#!/usr/bin/env bash
# EXP-024 pod 入口（3×H200）：stage 解压到 /data/stage 后 `setsid nohup bash /data/stage/run_all.sh > /data/run.log 2>&1 < /dev/null &`
# HF token 在 /data/.hf_token（scp 进来）。结果推 HF 私有仓库 alllwang/sn120-exp024。
set -uo pipefail
W=/data; S=$W/stage; R=$W/results; M=$W/models
VT=$W/venvT; VV=$W/venvV; PT=$VT/bin/python; PV=$VV/bin/python
AFF=$W/affine; KIT=$AFF/miner/kit; COMMIT=eaf1ad20e4d018c85b30f53e9807b2b9a6a75e25
HOTKEY=5C4hsyrmfBHWf2RHRxR4BrjDzNaHvgW38sFyZS8ky5S5K2Zq
export PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 KIT
mkdir -p $R $M $W/logs
stamp() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/progress.txt; }
waitfor() { while [ ! -e "$1" ]; do [ -e "$1.FAIL" ] && { stamp "dependency failed: $1"; return 1; }; sleep 20; done; }
stamp START; cp $S/data/*.json $R/ 2>/dev/null; rm -f $R/urls.json

SERVE="--served-model-name m --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching --max-num-seqs 512 --limit-mm-per-prompt {\"image\":0,\"video\":0}"   # = validator/probe/serve.sh
serve() {  # serve GPU PORT MODELDIR TAG -> 等到起来
  mkdir -p $W/vc$1/vllm $W/vc$1/triton $W/vc$1/inductor
  CUDA_VISIBLE_DEVICES=$1 VLLM_CACHE_ROOT=$W/vc$1/vllm TRITON_CACHE_DIR=$W/vc$1/triton TORCHINDUCTOR_CACHE_DIR=$W/vc$1/inductor VLLM_NO_USAGE_STATS=1 setsid nohup $VV/bin/vllm serve $3 --port $2 $SERVE > $W/logs/vllm-$4.log 2>&1 < /dev/null &
  echo $! > $W/vllm$1.pid
  for i in $(seq 600); do kill -0 $(cat $W/vllm$1.pid) 2>/dev/null || break; curl -sf http://127.0.0.1:$2/v1/models >/dev/null && { stamp "SERVE_UP $4 gpu$1"; return 0; }; sleep 2; done
  stamp "SERVE_FAIL $4"; return 1; }
unserve() { kill $(cat $W/vllm$1.pid) 2>/dev/null; sleep 15; pkill -9 -f "vllm serve.*--port $2" 2>/dev/null; sleep 5; }
ev() {  # ev PORT TOKDIR ROUND ROLE SEED EPOCH OUT
  $PV $S/evalrun.py --url http://127.0.0.1:$1 --tok $2 --round $S/data/$3 --role $4 --seed $5 --epoch $6 --out $R/$7.json >> $R/progress.txt 2>> $W/logs/$7.err \
    && stamp "EVAL_OK $7" || stamp "EVAL_FAIL $7"; }
train() {  # train GPU HEADDIR OUTTAG
  CUDA_VISIBLE_DEVICES=$1 TRITON_CACHE_DIR=$W/triton$1 $PT $KIT/train_step.py --head $2 --ref $M/ref --batch $S/data/batch600.json --hotkey $HOTKEY \
    --out $M/$3 --device cuda --report $R/step_$3.json > $W/logs/train-$3.log 2>&1 && { tail -1 $W/logs/train-$3.log >> $R/progress.txt; touch $M/$3.OK; stamp "TRAIN_OK $3"; } \
    || { touch $M/$3.OK.FAIL; stamp "TRAIN_FAIL $3"; tail -20 $W/logs/train-$3.log >> $R/progress.txt; }; }


# run2：run_all 的 vllm serve 少了 --limit-mm-per-prompt（线上 serve.sh 有），GPU0/1 失败；本脚本接管所有评测。
stamp RUN2_START
# GPU0
( serve 0 8000 $M/2cf21ce5 H || exit
  ev 8000 $M/2cf21ce5 round600.json head prod 600 H_Dp_prod &
  ev 8000 $M/2cf21ce5 round600.json head s1 600 H_Dp_s1 & wait
  ev 8000 $M/2cf21ce5 round600.json cand s1 600 C0_Dp_s1
  unserve 0 8000; touch $W/GPU0_DONE ) &
# GPU1
( serve 1 8001 $M/2ba4137e C1 || exit
  ev 8001 $M/2ba4137e round600.json cand prod 600 C1_Dp_prod &
  ev 8001 $M/2ba4137e round600.json cand s1 600 C1_Dp_s1 & wait
  unserve 1 8001
  waitfor $M/C4.OK && serve 1 8001 $M/C4 C4 && ev 8001 $M/2ba4137e round607.json cand prod 607 C4_D607_prod; unserve 1 8001
  touch $W/GPU1_DONE ) &
# GPU2：等 run_all 起的 C2c 训练进程结束
( P=$(pgrep -f "train_step.py --head /data/models/2cf21ce5" | head -1)
  while [ -n "$P" ] && kill -0 $P 2>/dev/null; do sleep 20; done
  if grep -q '"status": "updated"' $W/logs/train-C2c.log; then tail -1 $W/logs/train-C2c.log >> $R/progress.txt; touch $M/C2c.OK; stamp "TRAIN_OK C2c"; else touch $M/C2c.OK.FAIL; stamp TRAIN_FAIL C2c; tail -20 $W/logs/train-C2c.log >> $R/progress.txt; fi
  train 2 $M/2ba4137e C4
  [ -e $M/C2c.OK ] && serve 2 8002 $M/C2c C2c && ev 8002 $M/2cf21ce5 round600.json cand prod 600 C2c_Dp_prod; unserve 2 8002
  touch $W/GPU2_DONE ) &
( waitfor $M/C2c.OK && $PT $S/cmp_weights.py --head $M/2cf21ce5 --a $M/2ba4137e --b $M/C2c --out $R/weights_C1_vs_C2c.json >> $R/progress.txt 2>&1 && stamp CMP_OK ) &
wait
stamp ALL_DONE; touch $R/ALL_DONE
