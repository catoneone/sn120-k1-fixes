#!/usr/bin/env bash
# 第三轮（有门槛）：fp32 主权重 + 候选集归一化 + OPT；每步训一个候选 → 官方式门槛（1,000 题 × 4，z>0 且打满<20%）→ 接受或连主权重一起回滚
# 用法：bash gated.sh GPU NAME OPT（OPT = normgrad | sign）
set -uo pipefail
G=$1; NAME=$2; OPT=$3
W=/data; S=$W/stage; PY=$W/venvA/bin/python; R=$W/results; M=$W/models; SRC=$(cat $W/SRC)
export CUDA_VISIBLE_DEVICES=$G TRITON_CACHE_DIR=/data/triton$G VLLM_CACHE_ROOT=/data/vllm_cache$G PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "$(date -u +%FT%TZ) gpu$G $*" | tee -a $R/progress.txt; }
COMMON="--source $SRC --data $W/k1 --refdir $W/refcache --ref $M/base"
cd /tmp
$PY $S/bf16_exp.py stream $COMMON --host $S/replay/host_renorm --arm fp32_renorm --opt $OPT --start $M/7fa0305a --turns ${TURNS:-197-210} \
   --probe-every 100 --gate-tasks $S/gate.json --cand-dir /data/cand_$NAME --out $R/gated_$NAME --save $M/gated_$NAME > $R/gated_$NAME.out 2>&1
stamp GATED_${NAME}_EXIT $?
cd $S
$W/venvB/bin/python gen_heldout.py --ckpt $M/gated_$NAME --tasks $S/heldout.json --out $R/gen_gated_$NAME --k 1 > $R/gen_gated_$NAME.out 2>&1; stamp GEN_gated_${NAME}_EXIT $?
touch $R/DONE_$NAME
