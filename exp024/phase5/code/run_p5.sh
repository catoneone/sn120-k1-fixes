#!/usr/bin/env bash
# EXP-024 第五阶段 pod 入口（2×H200）：stage5 解压到 /data/stage5，`setsid nohup bash /data/stage5/run_p5.sh > /data/run.log 2>&1 < /dev/null &`
# R1 线上 kernel 本地训一步（L）、R2 kit 默认训一步（K），都带主权重，比线上候选 1b70bca8 的主权重；
# R3 同一损失下 kit / live / live_fusednorm / fp32ref 四种实现的梯度。大文件都放 /dev/shm。
set -uo pipefail
W=/data; S=$W/stage5; R=$W/results/phase5; M=$W/models; X=/dev/shm/p5
VT=$W/venvT; PT=$VT/bin/python
AFF=$W/affine; KIT=$AFF/miner/kit; COMMIT=9935e14
HOTKEY=$(python3 -c "import json;print(json.load(open('$S/data/turn718.json'))['turn']['hotkey'])")
HID=$(python3 -c "import json;print(json.load(open('$S/data/urls.json'))['58bcc262']['id'])")
CID=$(python3 -c "import json;print(json.load(open('$S/data/urls.json'))['1b70bca8']['id'])")
H=$M/58bcc262
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1 KIT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R $M $X $W/logs; ulimit -n 65536
stamp() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/progress.txt; }
stamp START
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d $AFF/.git ] || git clone -q https://github.com/AffineFoundation/affine.git $AFF
(cd $AFF && git fetch -q --depth 50 origin main; git checkout -q $COMMIT) && stamp "SOURCE $(cd $AFF && git rev-parse HEAD)" || { stamp SOURCE_FAIL; exit 1; }
uv venv -q $VT --python 3.12 && VIRTUAL_ENV=$VT uv pip install -q -r $AFF/miner/requirements-trainer.txt accelerate >> $R/progress.txt 2>&1 || { stamp VENV_FAIL; exit 1; }
cat > $W/dl.py <<'EOF'
import json, hashlib, sys, urllib.request, concurrent.futures as cf, pathlib, time
u = json.load(open('/data/stage5/data/urls.json'))[sys.argv[1]]; out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
def dl(job):
    name, url, sha = job; p = out / name
    for attempt in range(5):
        try:
            h = hashlib.sha256(); tmp = p.with_suffix(p.suffix + '.part')
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'exp024/5'}), timeout=1800) as r, tmp.open('wb') as f:
                for b in iter(lambda: r.read(16 << 20), b''): h.update(b); f.write(b)
            if h.hexdigest() != sha: raise IOError('sha mismatch')
            tmp.rename(p); return name
        except Exception as e:
            print('retry', name, e, flush=True); time.sleep(5)
    raise SystemExit('download failed ' + name)
with cf.ThreadPoolExecutor(10) as ex: list(ex.map(dl, [(n, f['url'], f['sha256']) for n, f in u['files'].items()]))
print('CKPT_OK', sys.argv[1], flush=True)
EOF
python3 $W/dl.py 58bcc262 $H >> $R/progress.txt 2>&1 &
(cd $KIT && $PT fetch_fp32_master.py --head $HID --out $H-fp32-master >> $R/progress.txt 2>&1) &
(cd $KIT && $PT fetch_fp32_master.py --head $CID --out $M/gate-master >> $R/progress.txt 2>&1) &
(cd $KIT && $PT fetch_head.py --out $M/ref --reference >> $R/progress.txt 2>&1) &
wait
[ -f $H/config.json ] && [ -f $H-fp32-master/FP32-MASTER.json ] && [ -d $M/gate-master ] && [ -f $M/ref/CHECKPOINT.json ] || { stamp SETUP_FAIL; exit 1; }
stamp SETUP_OK
cat > $W/hfpush.py <<'EOF'
import os, time, sys
from huggingface_hub import HfApi
api = HfApi(token=open('/data/.hf_token').read().strip())
while True:
    try: api.upload_folder(folder_path=sys.argv[1], path_in_repo='phase5', repo_id='alllwang/sn120-exp024', repo_type='dataset', ignore_patterns=['*.safetensors'])
    except Exception as e: print('push failed', e, flush=True)
    if os.path.exists(os.path.join(sys.argv[1], 'ALL_DONE')): break
    time.sleep(120)
EOF
setsid nohup $PT $W/hfpush.py $R > $W/logs/hfpush.log 2>&1 < /dev/null &

step() {  # step GPU TAG SCRIPT
  CUDA_VISIBLE_DEVICES=$1 TRITON_CACHE_DIR=$W/triton$1 $PT $3 --head $H --ref $M/ref --batch $S/data/batch718.json --hotkey $HOTKEY \
    --out $X/$2 --device cuda --activation-checkpointing --report $R/step_$2.json > $W/logs/train-$2.log 2>&1 \
    && { tail -1 $W/logs/train-$2.log >> $R/progress.txt; stamp "TRAIN_OK $2"; } || { stamp "TRAIN_FAIL $2"; tail -8 $W/logs/train-$2.log >> $R/progress.txt; }; }
probe() {  # probe GPU MODE
  CUDA_VISIBLE_DEVICES=$1 TRITON_CACHE_DIR=$W/triton$1 $PT $S/grad_probe.py --head $H --batch $S/data/batch718.json --hotkey $HOTKEY \
    --mode $2 --n 24 --out $X/grads_$2.safetensors > $W/logs/probe-$2.log 2>&1 \
    && { cp $X/grads_$2.safetensors.json $R/probe_$2.json; stamp "PROBE_OK $2"; } || { stamp "PROBE_FAIL $2"; tail -8 $W/logs/probe-$2.log >> $R/progress.txt; }; }
( step 0 L $S/train_live.py; probe 0 fp32ref; probe 0 live_fusednorm ) & P0=$!
( step 1 K $KIT/train_step.py; probe 1 kit; probe 1 live ) & P1=$!
wait $P0 $P1
for b in L K; do [ -d $X/$b/fp32-master ] && $PT $S/cmp_master.py --head-master $H-fp32-master --a $M/gate-master --b $X/$b/fp32-master --out $R/m_gate_vs_$b.json >> $R/progress.txt 2>&1; done
[ -d $X/L/fp32-master ] && [ -d $X/K/fp32-master ] && $PT $S/cmp_master.py --head-master $H-fp32-master --a $X/L/fp32-master --b $X/K/fp32-master --out $R/m_L_vs_K.json >> $R/progress.txt 2>&1
$PT $S/cmp_grads.py --dir $X --pairs kit:fp32ref live:fp32ref kit:live live_fusednorm:fp32ref live_fusednorm:live kit:live_fusednorm --out $R/cmp_grads.json >> $R/progress.txt 2>&1
stamp ALL_DONE; touch $R/ALL_DONE
