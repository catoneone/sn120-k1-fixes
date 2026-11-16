"""CPU test of v2-trainer-relax.diff on a tiny random Qwen3 model (no GPU, a few minutes).

Checks:
  1) minibatches = 1: the patched trainer writes the same LoRA as the official one, bit for bit (steps 2);
  2) minibatches > 1 runs, with steps x minibatches updates (and one per row at minibatches = 8000);
  3) a miner-set global_norm_cap is applied;
  4) out-of-bound values are rejected with the usual bound message;
  5) target_modules accepts in_proj_qkv / in_proj_z / out_proj and rejects in_proj_a;
  6) the new file caps pass the Params checks.

Setup (ROOT is any scratch directory; AFFINE is a checkout of the v2 branch):
  mkdir -p $ROOT/src $ROOT/src_patched $ROOT/tiny/tok
  cp -r $AFFINE/v2/sn120_v2 $ROOT/src/
  cp -r $AFFINE/v2/sn120_v2 $ROOT/src_patched/
  (cd $ROOT/src_patched && patch -p2 < v2-trainer-relax.diff)
  huggingface-cli download Qwen/Qwen3-0.6B vocab.json merges.txt tokenizer_config.json --local-dir $ROOT/tiny/tok
  python -I test_v2_trainer_relax.py $ROOT

Needs torch, transformers (with Qwen3) and safetensors. We ran it with torch 2.14.1 (CPU) and transformers 4.57.1.
The tiny model has no linear-attention layers, so check 5 covers the recipe bound only, not training those layers.
"""
import json, sys, shutil, math, importlib.util
from pathlib import Path
import torch, transformers
from safetensors.torch import load_file
root = Path(sys.argv[1]); work = root / 'tinytest' / 'work2'
shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
tok = transformers.AutoTokenizer.from_pretrained(str(root / 'tiny/tok'))
cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=32, tie_word_embeddings=False)
cfg.architectures = ['Qwen3ForCausalLM']
torch.manual_seed(0); transformers.Qwen3ForCausalLM(cfg).save_pretrained(str(work / 'base'), safe_serialization=True); tok.save_pretrained(str(work / 'base'))

def load(tree):
    for k in [k for k in sys.modules if k.startswith('sn120_v2')]: del sys.modules[k]
    sys.path.insert(0, str(root / tree))
    import sn120_v2.params as P, sn120_v2.gpu.train_job as T, sn120_v2.gpu.batch as B
    sys.path.pop(0)
    return P, T, B

groups = [(f'What is {i}+{i}?', [(f'<think>\nadd</think>\nThe answer is \\boxed{{{2*i}}}', 1.0), (f'<think>\nguess</think>\nThe answer is \\boxed{{{2*i+1}}}', 0.0)]) for i in range(12)]
def job(P, B, train, steps=2):
    base = P.Recipe(rank=8, alpha=16.0, lora_lr=1e-3, steps=steps, support_top_k=len(tok), support_top_p=1.0)
    r = base.for_miner(train) if train else base
    return {'base_dir': str(work / 'base'), 'base_hash': 'x', 'dtype': 'float32', 'recipe': r.to_dict(), 'seed': 7, 'rows': B.training_rows(groups)}
def lora(d): return load_file(str(d / 'adapter_model.safetensors'))

P0, T0, B0 = load('src')
r0 = T0.run(job(P0, B0, None), work / 'off', ['cpu'])
P1, T1, B1 = load('src_patched')
r1 = T1.run(job(P1, B1, None), work / 'new', ['cpu'])
a, b = lora(work / 'off'), lora(work / 'new')
same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
print('1) minibatches=1 bitwise equal to official:', same, round(r0['step_norm'], 6), round(r1['step_norm'], 6)); assert same

r2 = T1.run(job(P1, B1, {'minibatches': 4}), work / 'mb', ['cpu']); r2b = T1.run(job(P1, B1, {'minibatches': 8000}), work / 'mb_all', ['cpu']); print('2b) minibatches 8000 on 24 rows: updates', len(json.loads((work / 'mb_all/train_report.json').read_text())['grad_norms']))
rep = json.loads((work / 'mb/train_report.json').read_text())
print('2) steps=2 x minibatches=4: updates', len(rep['grad_norms']), 'step_norm', round(r2['step_norm'], 5), 'recipe_values.minibatches', rep['recipe_values']['minibatches'])
assert len(rep['grad_norms']) == 8

r3 = T1.run(job(P1, B1, {'minibatches': 4, 'global_norm_cap': 0.05}), work / 'cap', ['cpu'])
print('3) miner cap 0.05: before', round(r3['step_norm_before_cap'], 5), 'after', round(r3['step_norm'], 5)); assert abs(r3['step_norm'] - 0.05) < 1e-9

from sn120_v2.errors import BatchRejected
for bad in ({'minibatches': 8001}, {'minibatches': 0}, {'global_norm_cap': 7.0}, {'global_norm_cap': 0.001}, {'minibatches': 2.5}):
    try: P1.Recipe().for_miner(bad); print('4) NOT rejected', bad); raise SystemExit(1)
    except BatchRejected as e: print('4) rejected', bad, '->', str(e)[:60])
print('default recipe cap', P1.Recipe().global_norm_cap, 'minibatches', P1.Recipe().minibatches)
print('ALL OK')
r = P1.Recipe().for_miner({'target_modules': ['q_proj', 'in_proj_qkv', 'in_proj_z', 'out_proj']})
print('5) target_modules with linear attention accepted:', r.target_modules)
try: P1.Recipe().for_miner({'target_modules': ['in_proj_a']}); raise SystemExit('in_proj_a should be rejected')
except BatchRejected: print('5) in_proj_a rejected')
pp = P1.Params()
print('6) caps', pp.max_prompts, pp.max_rollouts, pp.max_batch_bytes // 2**20, 'MiB', pp.max_train_token_steps)
print('ALL OK 2')
