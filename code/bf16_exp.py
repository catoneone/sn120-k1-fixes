#!/usr/bin/env python3
"""EXP-023：K=1 一步更新的 bf16 舍入。同一个官方梯度，三种处理：
  rtn   官方现状：fp32 里算好这一步，四舍五入回 bf16（round-nearest-even），不留 fp32
  fp32  保留 fp32 主权重：对外发布的仍是 bf16 = rtn(主权重)，但下一步从主权重继续
  sr    随机舍入回 bf16（期望无偏，不留状态）

全部基于官方 /k1/replay/replay_k1.py（同一套 rows / 参考 pass / 梯度 pass / 全局裁剪 / fresh-AdamW 步 / 每张量相对封顶）。

子命令
  refs    算并缓存各轮的 KL 参考 logprob（只依赖 base 和批次，和 actor 无关）
  stage1  单步：复算一轮的官方梯度，比较三种处理：权重层面；一阶输出效应（固定文本上 ∇F·Δ）；RTN 的直接前向核对
  stream  多步：按顺序用各轮批次连续训练，arm = rtn | fp32 | fp32_renorm | sr

固定文本 X：e196 的批次（64 条，30a129a9 采样，不在训练流里）。F（每条按 token 平均后再按类别平均）：
  ll_correct / ll_wrong / ll_capped   采样 token 的 log p（全词表）
  logPS / logPS_capped                log P(候选集)，候选集 = top-k 20 → top-p 0.95（采样器规则），越负 = 尾部越大
  eos_think                           思考段（</think> 之前）每个位置全词表下 EOS 的概率之和的平均
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EOS = (248046, 248044)
TOPK, TOPP = 20, 0.95
SUPPORT = dict(top_k=TOPK, top_p=TOPP, temperature=1.0)
P = 'model.language_model.'
SLICES = {P + 'layers.31.mlp.down_proj.weight': None, P + 'layers.31.self_attn.o_proj.weight': None,
          P + 'layers.14.linear_attn.out_proj.weight': None,
          'lm_head.weight[248000:248077]': ('lm_head.weight', 248000, 248077),
          'lm_head.weight[100000:101024]': ('lm_head.weight', 100000, 101024)}
FNAMES = ('ll_correct', 'll_wrong', 'll_capped', 'logPS', 'logPS_capped', 'eos_think')
CH = 1 << 25            # flat chunk size for the per-tensor step math (bounded temporaries on the 1e9-element embeddings)
PROBE_BUDGET = 16384    # packed tokens per probe bin
bf16_w32 = lambda n, p, sl: p.detach().view(-1)[sl].float()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def say(log, *a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if log:
        with open(log, 'a') as f:
            f.write(s + '\n')


class Env:
    def __init__(self, source, host):
        sys.path.insert(0, str(source)); sys.path.insert(0, str(host))
        import torch
        self.torch = torch
        self.R = load_mod('replay_k1', HERE / 'replay' / 'replay_k1.py')
        self.B = load_mod('v824_batched', Path(host) / 'v8211_batched.py')
        self.FK = load_mod('v83_fast_kernels', Path(host) / 'v83_fast_kernels.py')
        from subnet.mimo_grpo.config import TrainerConfig, TRAIN_UNRESOLVED
        from subnet.mimo_grpo.data import InputBinding
        from subnet.mimo_grpo.projection import project_document
        from subnet.complete_group_codec import loads
        from subnet.complete_group_native_receipts import expand_native_grades
        self.TrainerConfig, self.TRAIN_UNRESOLVED, self.InputBinding = TrainerConfig, TRAIN_UNRESOLVED, InputBinding
        self.project_document, self.loads, self.expand_native_grades = project_document, loads, expand_native_grades
        self.vocab = None
        self.adv_pos = False        # ablation: keep only positive advantages
        self.no_overlong = False    # ablation: overlong_penalty = 0 (capped rollouts score 0 like wrong ones)
        self.opt = 'sign'           # 'sign' (official fresh-AdamW step) | 'normgrad' (gradient direction, same per-tensor step norm)
        FK, LC = self.FK, self.R.LOGIT_CHUNK

        def blp(runtime, items, *, gradients, engine, logit_chunk=2048, support=None):
            kw = {} if support is None else dict(support=support)
            return FK.packed_log_probs(runtime, items, gradients=gradients, logit_chunk=LC, **kw)
        self.B.batched_log_probs = blp

    # ------------------------------------------------------------------ batches
    def load_turn(self, data, e, eos=EOS):
        vocab = self.vocab
        turn = json.loads((Path(data) / 'turns' / f'{e}.json').read_text())
        tr = turn['training']
        settings = dict(tr['config']['settings'])
        if self.no_overlong:
            settings['overlong_penalty'] = 0.0
        cfg = self.TrainerConfig(**settings)
        inp = tr['input']
        binding = self.InputBinding(inp['run'], inp['epoch'], inp['checkpoint'], inp['contract_sha256'], inp['optimizer_step'])
        groups = []
        for row in turn['dispatch']['selected']:
            d = row['document']
            raw = (Path(data) / 'docs' / (d['sha256'] + '.json')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == d['sha256'], 'document hash'
            doc = self.loads(raw, len(raw))
            grade = row['native_grades']
            if grade.get('version') == 'complete-group-native-grading-compact-v1':
                grade = self.expand_native_grades(grade, doc['batch'])
            g, _ = self.project_document(doc, expected_document_sha256=d['sha256'], binding=binding,
                                         trusted_prompt=grade['prompt'], native_results=grade['results'], vocab_size=vocab,
                                         eos_token_ids=list(eos), group_size=cfg.group_size,
                                         unresolved_failures=cfg.cap_policy == self.TRAIN_UNRESOLVED)
            groups.append(g)
        assert [g.task_id for g in groups] == [x['task_id'] for x in tr['admitted_groups']], 'task order'
        return SimpleNamespace(e=e, turn=turn, tr=tr, cfg=cfg, groups=groups, head=inp['checkpoint'],
                               candidate=turn['candidate']['id'], sgd=tr['optimizer']['sgd'])

    def layout(self, T):
        B = self.B
        B.LOSS_AGG_MODE = T.tr['loss_aggregation']['mode']; B.LOSS_SCALE = float(T.tr['loss_aggregation']['loss_scale'])
        B.DROP_UNRESOLVED = bool((T.tr.get('unresolved_filter') or {}).get('enabled'))
        agg, loss_scale, _ = B.loss_aggregation(T.cfg)
        rows = B.layout_drop_unresolved(T.groups, T.cfg)[0] if B.DROP_UNRESOLVED else B._layout(T.groups, T.cfg)
        if self.adv_pos:
            for r in rows:
                r['advantage'] = max(0.0, r['advantage'])
        active = [i for i, r in enumerate(rows) if r['active']]
        world = int(T.tr['microbatch'].get('world') or self.R.WORLD)
        assign, nbins, loads_ = self.R.plan(rows, active, world)
        return SimpleNamespace(rows=rows, active=active, assign=assign, agg=agg, loss_scale=loss_scale,
                               plan_ok=loads_ == T.tr['microbatch'].get('rank_tokens'))

    # ------------------------------------------------------------------ KL reference cache
    def ensure_refs(self, Ts, refdir, ref_model_dir, log=None):
        torch = self.torch
        refdir = Path(refdir); refdir.mkdir(parents=True, exist_ok=True)
        missing = [T for T in Ts if T.cfg.kl_coefficient > 0 and not (refdir / f'{T.e}.pt').exists()]
        if not missing:
            return
        ref = self.R.Runtime(ref_model_dir, False, missing[0].cfg.kl_reference_checkpoint); self.FK.install(ref.model)
        for T in missing:
            L = self.layout(T)
            self.B._reference_pass(L.rows, [b for r in L.assign for b in r], T.cfg, ref, engine='packed',
                                   logit_chunk=self.R.LOGIT_CHUNK)
            refs = {i: r['ref'] for i, r in enumerate(L.rows) if r.get('ref') is not None}
            tmp = refdir / f'{T.e}.pt.tmp{os.getpid()}'
            torch.save(refs, tmp); os.replace(tmp, refdir / f'{T.e}.pt')
            say(log, 'REF', T.e, len(refs))
        del ref; gc.collect(); torch.cuda.empty_cache()

    def load_refs(self, L, T, refdir):
        if T.cfg.kl_coefficient > 0:
            for i, r in self.torch.load(Path(refdir) / f'{T.e}.pt').items():
                L.rows[i]['ref'] = r

    # ------------------------------------------------------------------ gradient (official path)
    def gradient(self, actor, params, T, refdir):
        torch, B, R = self.torch, self.B, self.R
        L = self.layout(T); self.load_refs(L, T, refdir)
        total = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params}
        details, losses = {}, {}
        versions = B._versions(actor.model)
        for r, bins in enumerate(L.assign):
            if not bins:
                continue
            for p in actor.model.parameters():
                p.grad = None
            B._gradient_pass(actor, L.rows, bins, T.cfg, L.agg, L.loss_scale, len(L.active),
                             R.RankCollector(params, total, len(bins)), versions, details, losses,
                             engine='packed', logit_chunk=R.LOGIT_CHUNK)
        for p in actor.model.parameters():
            p.grad = None
        norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in total.values()]
        norm = float(torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float32))
        scale = torch.clamp(T.cfg.max_gradient_norm / (torch.tensor(norm) + 1e-6), max=1.)
        for g in total.values():
            g.mul_(scale.to(g.device))
        pub = {(t['task_id'], t['nonce']): t for t in T.tr['trajectories']}
        dl = [abs(d['loss'] - pub[(d['task_id'], d['nonce'])]['loss']) for d in details.values()
              if (d['task_id'], d['nonce']) in pub]
        info = dict(objective=math.fsum(losses[i] for i in sorted(losses)), objective_published=T.tr.get('objective_loss'),
                    preclip=norm, preclip_published=T.tr.get('preclip_gradient_norm'), plan_ok=L.plan_ok,
                    rollouts=len(L.rows), active=len(L.active), rollout_loss_max_abs_diff=max(dl) if dl else None,
                    support_outside_tokens=sum(d.get('support_outside_tokens') or 0 for d in details.values()),
                    loss_tokens=sum(d.get('loss_tokens') or 0 for d in details.values()))
        return total, info

    # ------------------------------------------------------------------ fresh-AdamW step (replay_k1 step 3 math)
    def delta(self, decay, w32, g, lr):
        """replay_k1 step-3 delta on a flat chunk; decay = (original tensor dim >= 2 and not a bias)."""
        torch = self.torch
        d = g / (g.abs() + torch.tensor(self.R.FRESH_ADAM_EPS, dtype=torch.float32, device=g.device))
        if decay:
            d = d + torch.tensor(self.R.FRESH_ADAM_WD, dtype=torch.float32, device=g.device) * w32
        return torch.tensor(lr, dtype=torch.float32, device=g.device) * d

    @staticmethod
    def decay(name, p):
        return p.dim() >= 2 and not name.endswith('bias')

    def chunks(self, name, p, total, w32_of):
        """Yield (c, sl, w_bf16_flat_chunk, w32_chunk, g_chunk) over flat CH-element chunks (bounded temporaries)."""
        W, G = p.detach().view(-1), total[name].view(-1)
        for c, s0 in enumerate(range(0, W.numel(), CH)):
            sl = slice(s0, min(s0 + CH, W.numel()))
            yield c, sl, W[sl], w32_of(name, p, sl), G[sl]

    def scales(self, params, total, sgd, w32_of):
        lr, cap, trc = sgd['sgd_lr'], sgd['sgd_update_norm_cap'], sgd.get('sgd_tensor_rel_cap')
        rr, dn2, qq, gn2 = {}, [], {}, {}
        with self.torch.no_grad():
            for name, p in params:
                q = wn2 = g2 = 0.0; dec = self.decay(name, p)
                for c, sl, w, w32, g in self.chunks(name, p, total, w32_of):
                    q += float(self.delta(dec, w32, g, lr).double().pow(2).sum()); wn2 += float(w32.double().pow(2).sum())
                    g2 += float(g.double().pow(2).sum())
                qq[name], gn2[name] = q, g2
                rt = 1.0
                if trc is not None and q > 0:
                    wn = math.sqrt(wn2)
                    rt = min(1.0, trc * (1 - 1e-6) * wn / math.sqrt(q)) if wn > 0 else 1.0
                rr[name] = rt; dn2.append(q * rt * rt)
        raw = math.sqrt(math.fsum(dn2)); cscale = min(1.0, cap / raw) if raw > 0 else 1.0
        # normgrad: d_t = |d_sign,t| * g_t / |g_t|  (stored as the multiplier on g)
        self.gscale = {n: (cscale * rr[n] * math.sqrt(qq[n]) / math.sqrt(gn2[n])) if gn2[n] > 0 else 0.0 for n in rr}
        return {n: cscale * v for n, v in rr.items()}, dict(raw_update_norm=raw, raw_update_norm_published=sgd.get('raw_update_norm'),
                                                            tensors_rel_capped=sum(v < 1.0 for v in rr.values()),
                                                            tensors_rel_capped_published=sgd.get('tensors_rel_capped'))

    def sr_round(self, x, seed):
        """fp32 -> bf16 stochastic rounding: add uniform 16-bit noise to the dropped mantissa bits, then truncate (exact)."""
        torch = self.torch
        gen = torch.Generator(device=x.device); gen.manual_seed(seed)
        bits = x.contiguous().view(torch.int32)
        noise = torch.randint(0, 1 << 16, x.shape, generator=gen, device=x.device, dtype=torch.int32)
        out = ((bits + noise) & -65536).view(torch.float32)        # -65536 == 0xFFFF0000 as int32
        out = torch.where(torch.isfinite(x), out, x)
        return out.to(torch.bfloat16)                                # exact: low 16 bits are zero

    def new_weights(self, decay, w32, g, scale, lr, mode, seed, name=None):
        d = self.gscale[name] * g if (self.opt == 'normgrad' and name is not None) else scale * self.delta(decay, w32, g, lr)
        new32 = w32 - d
        newbf = self.sr_round(new32, seed) if mode == 'sr' else new32.to(self.torch.bfloat16)
        return d, new32, newbf


# ---------------------------------------------------------------------- fixed-text functionals F
def build_probe_items(env, data, e):
    T = env.load_turn(data, e)
    L = env.layout(T)
    items, cats = [], []
    for r in L.rows:
        ro = r['rollout']
        cat = 2 if ro.stop == 'max_tokens' else (0 if r['reward'] >= 0.5 else 1)
        items.append((list(r['group'].prompt), list(ro.tokens))); cats.append(cat)
    return items, cats


def probe_bins(items, budget=PROBE_BUDGET):
    order = sorted(range(len(items)), key=lambda i: -(len(items[i][0]) + len(items[i][1])))
    bins = []
    for i in order:
        n = len(items[i][0]) + len(items[i][1])
        for b in bins:
            if b[0] + n <= budget:
                b[0] += n; b[1].append(i); break
        else:
            bins.append([n, [i]])
    return [sorted(b[1]) for b in bins]


def probe_sums(env, runtime, items, cats, think_end, idx, *, grad_of=None, counts=None):
    """Per-rollout-mean functionals over one packed bin. grad_of=k: backward F_k's share (F = mean over class of per-rollout means)."""
    torch = env.torch
    from torch.utils.checkpoint import checkpoint
    model, device = runtime.model, runtime.device
    sub = [items[i] for i in idx]
    seqs = [p + o for p, o in sub]; lens = [len(s) for s in seqs]
    with torch.set_grad_enabled(grad_of is not None):
        ids = torch.tensor([t for s in seqs for t in s], device=device)[None]
        pos = torch.cat([torch.arange(n) for n in lens])[None].to(device)
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32); cu[1:] = torch.tensor(lens).cumsum(0)
        offsets = cu.tolist(); cu = cu.to(device)
        hidden = model.model(ids, position_ids=pos, use_cache=False, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                             max_length_q=max(lens), max_length_k=max(lens),
                             attention_mask={'full_attention': None, 'linear_attention': None}).last_hidden_state[0]
        # per output token: which rollout (local), class, inside-think flag, per-rollout weights for each functional
        rid, think = [], []
        for r, (p, o) in enumerate(sub):
            end = o.index(think_end) if think_end in o else len(o)
            rid += [r] * len(o); think += [1.0 if j <= end else 0.0 for j in range(len(o))]
        pos_idx = torch.cat([torch.arange(offsets[r] + len(p) - 1, offsets[r] + len(p) + len(o) - 1) for r, (p, o) in enumerate(sub)])
        states = hidden.index_select(0, pos_idx.to(device))
        targets = torch.tensor([t for _, o in sub for t in o], device=device)
        rid_t = torch.tensor(rid, device=device); think_t = torch.tensor(think, device=device)
        # weight per token for each functional = 1 / (len_r * n_class) so that sum = class mean of per-rollout means
        W = torch.zeros(len(FNAMES), len(rid), device=device)
        for r, i in enumerate(idx):
            c, n_tok = cats[i], len(items[i][1])
            sel = rid_t == r
            if c in (0, 1, 2):
                k = c
                W[k, sel] = 1.0 / (n_tok * counts['cat'][c])
            W[3, sel] = 1.0 / (n_tok * counts['all'])
            if c == 2:
                W[4, sel] = 1.0 / (n_tok * counts['cat'][2])
            nt = float(think_t[sel].sum())
            if nt > 0:
                W[5, sel] = think_t[sel] / (nt * counts['think'])

        def terms(h, t, w):
            logits = model.lm_head(h).float()
            lse = torch.logsumexp(logits, dim=-1)
            logp = logits.gather(-1, t[:, None]).squeeze(-1) - lse
            values, _ = logits.topk(TOPK, dim=-1)
            with torch.no_grad():
                pr = torch.softmax(values, dim=-1)
                keep = (pr.cumsum(-1) - pr) < TOPP
            lps = torch.logsumexp(values.masked_fill(~keep, float('-inf')), dim=-1) - lse
            peos = torch.exp(logits[:, list(EOS)] - lse[:, None]).sum(-1)
            per = torch.stack([logp, logp, logp, lps, lps, peos])
            return (per * w).sum(-1)
        out = torch.zeros(len(FNAMES), device=device, dtype=torch.float32)
        CH = 4096
        for s in range(0, len(targets), CH):
            h, t, w = states[s:s + CH], targets[s:s + CH], W[:, s:s + CH]
            out = out + (checkpoint(terms, h, t, w, use_reentrant=False) if grad_of is not None else terms(h, t, w))
        if grad_of is not None:
            out[grad_of].backward()
        return out.detach()


def probe_counts(items, cats, think_end):
    c = {0: 0, 1: 0, 2: 0}
    for x in cats:
        c[x] += 1
    return dict(cat={k: max(v, 1) for k, v in c.items()}, all=len(items),
                think=sum(1 for p, o in items if (o.index(think_end) if think_end in o else len(o)) >= 0))


def probe_values(env, runtime, items, cats, think_end):
    counts = probe_counts(items, cats, think_end)
    tot = None
    for idx in probe_bins(items):
        v = probe_sums(env, runtime, items, cats, think_end, idx, counts=counts)
        tot = v if tot is None else tot + v
    return {k: float(x) for k, x in zip(FNAMES, tot.tolist())}


def slice_of(name_key, sd_get):
    spec = SLICES[name_key]
    if spec is None:
        return sd_get(name_key)
    n, a, b = spec
    return sd_get(n)[a:b]


# ---------------------------------------------------------------------- stage 1
def model_cfg(env, d):
    c = json.loads((Path(d) / 'config.json').read_text())
    env.vocab = (c.get('text_config') or c)['vocab_size']
    tok = json.loads((Path(d) / 'tokenizer.json').read_text())['added_tokens']
    return next(t['id'] for t in tok if t['content'] == '</think>')


def cmd_stage1(a):
    env = Env(a.source, a.host); torch = env.torch
    think_end = model_cfg(env, a.head)
    log = Path(a.out) / 'log.txt'; Path(a.out).mkdir(parents=True, exist_ok=True)
    T = env.load_turn(a.data, a.turn)
    env.ensure_refs([T], a.refdir, a.ref, log)
    actor = env.R.Runtime(a.head, True, T.head); env.FK.install(actor.model)
    params = [(n, p) for n, p in actor.model.named_parameters() if p.requires_grad]
    total, info = env.gradient(actor, params, T, a.refdir)
    say(log, 'GRAD', json.dumps(info))
    lr = T.sgd['sgd_lr']
    sc, sinfo = env.scales(params, total, T.sgd, bf16_w32)
    say(log, 'SCALES', json.dumps(sinfo))
    cand = None
    if a.candidate:
        from safetensors import safe_open
        idx = json.loads(Path(a.candidate, 'model.safetensors.index.json').read_text())['weight_map']; handles = {}

        def cand(name):
            f = idx[name]
            if f not in handles:
                handles[f] = safe_open(str(Path(a.candidate, f)), framework='pt', device='cpu')
            return handles[f].get_tensor(name)
    modes = ('fp32', 'rtn', 'sr')
    st = {m: dict(changed=0, d2=0.0, dot_int=0.0, g_dot=0.0) for m in modes}
    int2 = 0.0; numel = 0; mism = 0; small_changed = 0
    seed = lambda i, c: 1000003 * a.turn + 4096 * i + c

    def step_chunks(i, name, p):
        dec = env.decay(name, p)
        for c, sl, w, w32, g in env.chunks(name, p, total, bf16_w32):
            d, new32, rtn = env.new_weights(dec, w32, g, sc[name], lr, 'rtn', seed(i, c))
            yield c, sl, w, w32, g, d, rtn, env.sr_round(new32, seed(i, c))
    with torch.no_grad():
        for i, (name, p) in enumerate(params):
            cflat = cand(name).view(-1) if cand is not None else None
            numel += p.numel()
            for c, sl, w, w32, g, d, rtn, sr in step_chunks(i, name, p):
                dint = -d
                int2 += float(dint.double().pow(2).sum())
                for m, dm in (('fp32', dint), ('rtn', rtn.float() - w32), ('sr', sr.float() - w32)):
                    s = st[m]
                    s['changed'] += int((dm != 0).sum()); s['d2'] += float(dm.double().pow(2).sum())
                    s['dot_int'] += float((dm.double() * dint.double()).sum()); s['g_dot'] += float((g.double() * dm.double()).sum())
                small_changed += int(((rtn != w) & (w32.abs() < 2 ** -12)).sum())
                if cflat is not None:
                    mism += int((cflat[sl].to(rtn.device) != rtn).sum())
    res = dict(turn=a.turn, head=T.head, candidate=T.candidate, grad=info, scales=sinfo, numel=numel,
               intended_norm=math.sqrt(int2), candidate_mismatched_elements=mism if cand else None,
               rtn_changed_small_weights=small_changed,
               changed_elements_published=T.sgd.get('projected_changed_elements'))
    for m in modes:
        s = st[m]
        res[m] = dict(changed=s['changed'], changed_frac=s['changed'] / numel, norm=math.sqrt(s['d2']),
                      norm_ratio=math.sqrt(s['d2'] / int2), cos_intended=s['dot_int'] / math.sqrt(s['d2'] * int2) if s['d2'] else 0.,
                      first_order_loss_change=s['g_dot'])
    say(log, 'WEIGHTS', json.dumps({m: res[m] for m in modes}))
    # first-order effect on fixed text: dF_m = grad F . Delta_m (same Delta_m recomputed tensor by tensor)
    items, cats = build_probe_items(env, a.data, a.probe_turn)
    counts = probe_counts(items, cats, think_end)
    base_vals = probe_values(env, actor, items, cats, think_end)
    res['probe_head'] = base_vals
    lin = {m: {} for m in modes}
    for k, fname in enumerate(FNAMES):
        for p in actor.model.parameters():
            p.grad = None
        for idx in probe_bins(items):
            probe_sums(env, actor, items, cats, think_end, idx, grad_of=k, counts=counts)
        acc = {m: 0.0 for m in modes}
        with torch.no_grad():
            for i, (name, p) in enumerate(params):
                if p.grad is None:
                    continue
                GF = p.grad.view(-1)
                for c, sl, w, w32, g, d, rtn, sr in step_chunks(i, name, p):
                    gF = GF[sl].double()
                    acc['fp32'] += float((gF * (-d).double()).sum())
                    acc['rtn'] += float((gF * (rtn.float() - w32).double()).sum())
                    acc['sr'] += float((gF * (sr.float() - w32).double()).sum())
                p.grad = None
        for m in modes:
            lin[m][fname] = acc[m]
        say(log, 'LIN', fname, json.dumps({m: acc[m] for m in modes}))
    for p in actor.model.parameters():
        p.grad = None
    res['probe_linear_delta'] = lin
    # direct check: apply RTN for real, forward again, restore
    saved = {}
    with torch.no_grad():
        for i, (name, p) in enumerate(params):
            flat = p.data.view(-1); keep = []
            for c, sl, w, w32, g, d, rtn, sr in step_chunks(i, name, p):
                ch = (rtn != w).nonzero(as_tuple=True)[0]
                if len(ch):
                    keep.append((ch + sl.start, w[ch].clone()))
                    flat[sl].copy_(rtn)
            if keep:
                saved[name] = keep
    rtn_vals = probe_values(env, actor, items, cats, think_end)
    res['probe_direct_delta_rtn'] = {k: rtn_vals[k] - base_vals[k] for k in FNAMES}
    with torch.no_grad():
        for name, p in params:
            flat = p.data.view(-1)
            for ix, v in saved.get(name, []):
                flat[ix] = v
    say(log, 'DIRECT_RTN', json.dumps(res['probe_direct_delta_rtn']))
    (Path(a.out) / f'stage1_e{a.turn}.json').write_text(json.dumps(res, indent=1))
    say(log, 'STAGE1_DONE', a.turn)


# ---------------------------------------------------------------------- stage 2 stream
def save_like(start_dir, out_dir, model, log=None):
    """Write the actor weights with the start checkpoint's shard layout; copy every non-weight file."""
    import shutil
    from safetensors.torch import save_file
    start_dir, out_dir = Path(start_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    idx = json.loads((start_dir / 'model.safetensors.index.json').read_text())
    sd = model.state_dict()
    by_file = {}
    for k, f in idx['weight_map'].items():
        by_file.setdefault(f, []).append(k)
    missing = [k for k in idx['weight_map'] if k not in sd]
    if missing:
        raise SystemExit(f'save_like: {len(missing)} names not in state_dict, e.g. {missing[:3]}')
    for f, names in by_file.items():
        save_file({k: sd[k].detach().to('cpu').contiguous() for k in names}, str(out_dir / f), metadata={'format': 'pt'})
    for fp in start_dir.iterdir():
        if fp.is_file() and not fp.name.endswith('.safetensors') and not (out_dir / fp.name).exists():
            shutil.copy2(fp, out_dir / fp.name)
    say(log, 'SAVED', out_dir)


def gate_run(a, ckpt, out_path, log):
    """Run the official-style gate eval (gate_eval.py under the vLLM venv) on a checkpoint dir; returns per-sample rows."""
    import subprocess
    t0 = time.time()
    r = subprocess.run([a.gate_py, str(HERE / 'gate_eval.py'), '--ckpt', str(ckpt), '--tasks', a.gate_tasks, '--out', str(out_path),
                        '--k', str(a.gate_k)], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'gate eval failed ({r.returncode}): {r.stderr[-3000:]}')
    rows = [json.loads(x) for x in Path(out_path).read_text().splitlines() if x.strip()]
    say(log, 'GATE_EVAL', ckpt, f'{time.time() - t0:.0f}s', Path(str(out_path) + '.summary.json').read_text())
    return rows


def gate_compare(head, cand):
    """Paired per-task difference of mean correctness (common random numbers); official rule: z > 0 and capped < 20%."""
    from collections import defaultdict
    acc = defaultdict(lambda: [[], []])
    for k, rows in ((0, head), (1, cand)):
        for x in rows:
            acc[x['index']][k].append(float(x['correct']))
    d = [sum(c) / len(c) - sum(h) / len(h) for h, c in acc.values()]
    n = len(d); mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    z = mean / se if se > 0 else 0.0
    cap = sum(x['capped'] for x in cand) / len(cand)
    return dict(diff=mean, se=se, z=z, cand_pass=sum(x['correct'] for x in cand) / len(cand),
                head_pass=sum(x['correct'] for x in head) / len(head), cand_capped=cap, accept=bool(z > 0 and cap < 0.20))


def cmd_stream(a):
    env = Env(a.source, a.host); torch = env.torch
    think_end = model_cfg(env, a.start)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log = out / 'log.txt'
    env.adv_pos, env.no_overlong, env.opt = a.adv_pos, a.no_overlong, a.opt
    say(log, 'ARM', a.arm, 'opt', a.opt, 'adv_pos', a.adv_pos, 'no_overlong', a.no_overlong)
    lo, hi = map(int, a.turns.split('-'))
    Ts = [env.load_turn(a.data, e) for e in range(lo, hi + 1)]
    env.ensure_refs(Ts, a.refdir, a.ref, log)
    if a.arm == 'fp32_renorm':
        assert hasattr(env.B, 'SUPPORT_RENORM'), 'fp32_renorm needs the patched host'
        env.B.SUPPORT_RENORM = dict(SUPPORT)
    mode = 'sr' if a.arm == 'sr' else 'rtn'
    use_master = a.arm in ('fp32', 'fp32_renorm', 'randsign')
    actor = env.R.Runtime(a.start, True, Ts[0].head); env.FK.install(actor.model)
    params = [(n, p) for n, p in actor.model.named_parameters() if p.requires_grad]
    pd = dict(params)
    master = {n: p.detach().float().cpu().pin_memory() for n, p in params} if use_master else None
    sd_get = lambda n: pd[n].detach()
    start_slices = {k: slice_of(k, sd_get).float().cpu().clone() for k in SLICES}
    import numpy as np
    dirs = dict(np.load(a.dirs))
    items, cats = build_probe_items(env, a.data, a.probe_turn)

    def snapshot(step):
        vals = probe_values(env, actor, items, cats, think_end)
        al = {}
        for dn in ('collapse_b1', 'collapse_orig'):
            agree = n = 0
            agree_m = n_m = 0
            for k in SLICES:
                dv = torch.from_numpy(dirs[f'{dn}|{k}']).float()
                cur = slice_of(k, sd_get).float().cpu() - start_slices[k]
                m = (cur != 0) & (dv != 0)
                agree += int((torch.sign(cur[m]) == dv[m]).sum()); n += int(m.sum())
                if master is not None:
                    full = master[SLICES[k][0]][SLICES[k][1]:SLICES[k][2]] if SLICES[k] else master[k]
                    cm = full - start_slices[k]
                    mm = (cm != 0) & (dv != 0)
                    agree_m += int((torch.sign(cm[mm]) == dv[mm]).sum()); n_m += int(mm.sum())
            al[dn] = dict(published_agree=agree / max(n, 1), published_nonzero=n,
                          master_agree=(agree_m / max(n_m, 1)) if master is not None else None, master_nonzero=n_m)
        rec = dict(step=step, probe=vals, align=al)
        with open(out / 'snapshots.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        say(log, 'SNAP', step, json.dumps(vals), json.dumps({k: round(v['published_agree'], 4) for k, v in al.items()}))

    snapshot(0)
    cum_pub2 = 0.0
    gated = bool(a.gate_tasks)
    if gated:
        h0 = out / 'gate_s00.jsonl'
        head_rows = ([json.loads(x) for x in h0.read_text().splitlines() if x.strip()]
                     if Path(str(h0) + '.summary.json').exists() else gate_run(a, a.start, h0, log))   # reuse the start eval on restart
        prev_b = {n: p.detach().cpu().clone() for n, p in params}
        prev_m = {n: torch.empty_like(m) for n, m in master.items()} if master is not None else None
        accepted = 0
    for s, T in enumerate(Ts * a.passes, start=1):
        t0 = time.time()
        if gated:
            with torch.no_grad():
                for n, p in params:
                    prev_b[n].copy_(p.detach())
                    if prev_m is not None:
                        prev_m[n].copy_(master[n])
        if a.arm == 'randsign':
            # null control: same fresh-AdamW step, caps and fp32 master, but the gradient replaced by random signs
            total = {}
            for i, (n, p) in enumerate(params):
                gen = torch.Generator(device=p.device); gen.manual_seed(7919 * s + i)
                total[n] = torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=torch.int8).float().mul_(2).sub_(1)
            info = dict(objective=0.0, preclip=0.0, support_outside_tokens=0, loss_tokens=0)
        else:
            total, info = env.gradient(actor, params, T, a.refdir)
        w32_of = (lambda n, p, sl: master[n].view(-1)[sl].to(p.device, non_blocking=True)) if use_master else bf16_w32
        sc, sinfo = env.scales(params, total, T.sgd, w32_of)
        lr = T.sgd['sgd_lr']
        int2 = real2 = dot = g_int = g_real = 0.0; changed = 0
        with torch.no_grad():
            for i, (name, p) in enumerate(params):
                dec = env.decay(name, p); flat = p.data.view(-1)
                for c, sl, w, w32, g in env.chunks(name, p, total, w32_of):
                    d, new32, newbf = env.new_weights(dec, w32, g, sc[name], lr, mode, 1000003 * s + 4096 * i + c, name=name)
                    real = newbf.float() - w.float()
                    int2 += float(d.double().pow(2).sum()); real2 += float(real.double().pow(2).sum())
                    dot += float((real.double() * (-d).double()).sum())
                    g_int += float((g.double() * (-d).double()).sum()); g_real += float((g.double() * real.double()).sum())
                    changed += int((newbf != w).sum())
                    if master is not None:
                        master[name].view(-1)[sl].copy_(new32.cpu())
                    flat[sl].copy_(newbf)
        del total; gc.collect(); torch.cuda.empty_cache()
        cum_pub2 += real2
        rec = dict(step=s, turn=T.e, **info, **sinfo, intended_norm=math.sqrt(int2), realized_norm=math.sqrt(real2),
                   realized_over_intended=math.sqrt(real2 / int2) if int2 else None,
                   cos=dot / math.sqrt(int2 * real2) if int2 and real2 else None, changed=changed,
                   first_order_loss_change_intended=g_int, first_order_loss_change_realized=g_real, seconds=time.time() - t0)
        if gated:
            save_like(a.start, a.cand_dir, actor.model, log)
            cand_rows = gate_run(a, a.cand_dir, out / f'gate_s{s:02d}.jsonl', log)
            g = gate_compare(head_rows, cand_rows)
            if g['accept']:
                head_rows = cand_rows; accepted += 1
            else:
                with torch.no_grad():
                    for n, p in params:
                        p.data.copy_(prev_b[n].to(p.device))
                        if prev_m is not None:
                            master[n].copy_(prev_m[n])
            rec.update(gate=g, accepted_so_far=accepted, seconds=time.time() - t0)
            say(log, 'GATE', s, T.e, 'GREEN' if g['accept'] else 'RED', f"z {g['z']:+.2f} diff {g['diff']:+.4f} cand {g['cand_pass']:.4f} "
                f"head {g['head_pass']:.4f} capped {g['cand_capped']:.3f} accepted {accepted}/{s}")
        with open(out / 'steps.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        say(log, 'STEP', s, T.e, f"changed {changed} real/int {rec['realized_over_intended']:.3f} cos {rec['cos']:.3f} "
            f"preclip {info['preclip']:.4f} obj {info['objective']:.6f} outside {info['support_outside_tokens']} {rec['seconds']:.0f}s")
        if s % a.probe_every == 0 or s == len(Ts) * a.passes:
            snapshot(s)
    if a.save:
        save_like(a.start, a.save, actor.model, log)
    say(log, 'STREAM_DONE', a.arm)


def cmd_refs(a):
    env = Env(a.source, a.host); model_cfg(env, a.ref)
    lo, hi = map(int, a.turns.split('-'))
    env.ensure_refs([env.load_turn(a.data, e) for e in range(lo, hi + 1)], a.refdir, a.ref, Path(a.refdir) / 'log.txt')


def cmd_dry(a):
    """CPU check of the batch path: rebuild each turn, compare the planned rank tokens with the published plan."""
    env = Env(a.source, a.host); think_end = model_cfg(env, a.cfgdir)
    lo, hi = map(int, a.turns.split('-'))
    items, cats = build_probe_items(env, a.data, a.probe_turn)
    c = probe_counts(items, cats, think_end)
    print('PROBE', a.probe_turn, len(items), 'cats', c['cat'], 'tokens', sum(len(o) for _, o in items), 'bins', len(probe_bins(items)),
          'with </think>', sum(think_end in o for _, o in items), flush=True)
    for e in range(lo, hi + 1):
        T = env.load_turn(a.data, e); L = env.layout(T)
        cats = [2 if r['rollout'].stop == 'max_tokens' else (0 if r['reward'] >= 0.5 else 1) for r in L.rows]
        print('DRY', e, 'groups', len(T.groups), 'rows', len(L.rows), 'active', len(L.active), 'plan_ok', L.plan_ok,
              'cats', {c: cats.count(c) for c in (0, 1, 2)}, 'lr', T.sgd['sgd_lr'], 'rel_cap', T.sgd.get('sgd_tensor_rel_cap'),
              'kl', T.cfg.kl_coefficient, 'mode', T.tr['loss_aggregation']['mode'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['refs', 'stage1', 'stream', 'dry'])
    ap.add_argument('--source', required=True); ap.add_argument('--host', default=str(HERE / 'replay' / 'host'))
    ap.add_argument('--data', required=True); ap.add_argument('--refdir', default='/data/refcache')
    ap.add_argument('--ref'); ap.add_argument('--out')
    ap.add_argument('--turn', type=int); ap.add_argument('--head'); ap.add_argument('--candidate')
    ap.add_argument('--probe-turn', type=int, default=196)
    ap.add_argument('--turns'); ap.add_argument('--arm', choices=['rtn', 'fp32', 'fp32_renorm', 'sr', 'randsign'])
    ap.add_argument('--start'); ap.add_argument('--passes', type=int, default=1); ap.add_argument('--probe-every', type=int, default=6)
    ap.add_argument('--save'); ap.add_argument('--dirs', default=str(HERE / 'dirs.npz')); ap.add_argument('--cfgdir')
    ap.add_argument('--opt', choices=['sign', 'normgrad'], default='sign'); ap.add_argument('--adv-pos', action='store_true')
    ap.add_argument('--no-overlong', action='store_true')
    ap.add_argument('--gate-tasks'); ap.add_argument('--gate-py', default='/data/venvB/bin/python'); ap.add_argument('--gate-k', type=int, default=4)
    ap.add_argument('--cand-dir')
    a = ap.parse_args()
    dict(refs=cmd_refs, stage1=cmd_stage1, stream=cmd_stream, dry=cmd_dry)[a.cmd](a)


if __name__ == '__main__':
    main()
