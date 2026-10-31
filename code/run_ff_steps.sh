#!/usr/bin/env bash
# One step or a chain? Large free files (affine-batch-v1) from one head, four arms per file. See EXPERIMENT-ff-steps.md.
#
# Required env:
#   PY, PYV      training venv python / vLLM venv python (as in run_2x2.sh)
#   SRC          signed source tree (bundle a2a03cb3...)
#   BASE         MiMo base checkpoint (the KL reference)
#   CK           head checkpoint dir (bf16; also the tokenizer and chat template used to render the files)
#   FILES        space-separated paths of large free files (affine-batch-v1), ideally >= 100 prompts each
#   OUT          results dir
# Optional env:
#   MASTER       dir with the head's published fp32 master (*.safetensors). Without it the master is fp32(bf16 head).
#   HOST, HOST_RENORM   published /k1/replay/host/ and the same with patch/support_renorm.diff (default: built here)
#   GATE         gate task file (default data/gate_B.json); tasks whose problem text appears in a file are excluded per file
#   PER_STEP     prompts per chain sub-step (default 9)        SEED   chain order seed (default 0)
#   EXTRA        1 = also run the chain at 3x the sub-step size (granularity check)
#   GPUS         default "0 1 2 3 4 5 6 7"
set -uo pipefail
: "${PY:?}" "${PYV:?}" "${SRC:?}" "${BASE:?}" "${CK:?}" "${FILES:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
get() { curl -sSL --fail --retry 3 --max-time 180 -A sn120-replay/1 -o "$2" "https://affine.io/k1/replay/$1"; }
if [ ! -f "$HERE/replay/replay_k1.py" ]; then
  mkdir -p "$HERE/replay" && get replay_k1.py "$HERE/replay/replay_k1.py" || { echo 'cannot fetch replay_k1.py'; exit 1; }
fi
if [ -z "${HOST:-}" ]; then
  HOST=$HERE/replay/host
  if [ ! -f "$HOST/v8211_batched.py" ]; then
    mkdir -p "$HOST"; for f in v8211_batched.py v83_fast_kernels.py v83_dp.py; do get "host/$f" "$HOST/$f" || exit 1; done
  fi
fi
if [ -z "${HOST_RENORM:-}" ]; then
  HOST_RENORM=$HERE/replay/host_renorm
  if [ ! -f "$HOST_RENORM/v8211_batched.py" ]; then
    rm -rf "$HERE/replay/build" && mkdir -p "$HERE/replay/build" && cp -r "$HOST" "$HERE/replay/build/host"
    (cd "$HERE/replay/build" && patch -p1 -s < "$HERE/../patch/support_renorm.diff") || { echo 'patch failed'; exit 1; }
    mv "$HERE/replay/build/host" "$HOST_RENORM" && rm -rf "$HERE/replay/build"
  fi
fi
GATE=${GATE:-$HERE/../data/gate_B.json}; GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); NG=${#GPUS[@]}
PER_STEP=${PER_STEP:-9}; SEED=${SEED:-0}; REFS=$OUT/refcache; mkdir -p "$OUT" "$REFS"
MASTER_ARG=(); [ -n "${MASTER:-}" ] && MASTER_ARG=(--start-master "$MASTER")
stamp() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/progress.txt"; }
gpu_env() { export CUDA_VISIBLE_DEVICES=$1 TRITON_CACHE_DIR=$OUT/triton$1 VLLM_CACHE_ROOT=$OUT/vllm_cache$1 PYTHONUNBUFFERED=1 \
                   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; }
read -r -a FL <<< "$FILES"

# sub-steps per file after the live cap (125 prompts / 1,000 rollouts, whole prompts in file order)
substeps() { "$PY" -c "
import json, math, sys
g = json.load(open(sys.argv[1]))['groups']; k = n = 0
for x in g:
    if k == 125 or n + len(x['rollouts']) > 1000: break
    k += 1; n += len(x['rollouts'])
print(math.ceil(k / int(sys.argv[2])))" "$1" "$2"; }
obs4() { "$PY" -c "import sys; S=int(sys.argv[1]); print(','.join(str(x) for x in sorted({max(1, round(S*f)) for f in (0.25, 0.5, 0.75, 1.0)})))" "$1"; }

# 1. KL reference log-probs: one pass per file and grouping (whole file, chain), split over the GPUs
JOBS=()
for i in "${!FL[@]}"; do
  JOBS+=("F$((i+1)) ${FL[$i]} 0"); JOBS+=("F$((i+1)) ${FL[$i]} $PER_STEP")
  [ "${EXTRA:-0}" = 1 ] && JOBS+=("F$((i+1)) ${FL[$i]} $((PER_STEP * 3))")
done
for g in "${!GPUS[@]}"; do
  ( for j in "${!JOBS[@]}"; do (( j % NG == g )) || continue; read -r TAG F P <<< "${JOBS[$j]}"
      ( gpu_env "${GPUS[$g]}"; cd /tmp && "$PY" "$HERE/bf16_exp.py" refs --source "$SRC" --refdir "$REFS" --ref "$BASE" \
          --ff "$F" --ff-tok "$CK" --ff-tag "$TAG" --ff-per-step "$P" --ff-seed "$SEED" >> "$OUT/refs_${GPUS[$g]}.out" 2>&1 ) \
        || stamp "REFS_FAIL $TAG per_step=$P"
    done ) &
done; wait
stamp REFS_DONE "$(ls "$REFS"/*.pt 2>/dev/null | wc -l) files"

# 2. Start eval of the head on the gate set
( gpu_env "${GPUS[0]}"; cd "$HERE" && "$PYV" gate_eval.py --ckpt "$CK" --tasks "$GATE" --out "$OUT/start/gate.jsonl" > "$OUT/start.out" 2>&1 ) \
  && stamp START_EVAL "$(cat "$OUT/start/gate.jsonl.summary.json")" || { stamp START_EVAL_FAIL; exit 1; }

# 3. Arms: NAME FILE RECIPE PER_STEP OBS
#    live  = fp32 master + fresh-AdamW (sign) step, no renorm (the live v8.3.3f/h update)
#    ours  = fp32 master + support renorm + gradient-direction step at 1x
ARMS=()
for i in "${!FL[@]}"; do
  F=${FL[$i]}; T=F$((i+1)); S=$(substeps "$F" "$PER_STEP")
  ARMS+=("${T}_C_ours $F ours $PER_STEP $(obs4 "$S")")
  ARMS+=("${T}_C_live $F live $PER_STEP $(obs4 "$S")")
  ARMS+=("${T}_P_live $F live 0 1")
  ARMS+=("${T}_P_ours $F ours 0 1")
  if [ "${EXTRA:-0}" = 1 ]; then S3=$(substeps "$F" $((PER_STEP * 3))); ARMS+=("${T}_C3_ours $F ours $((PER_STEP * 3)) $(obs4 "$S3")"); fi
done
run_arm() {  # GPU NAME FILE RECIPE PER_STEP OBS
  local G=$1 NAME=$2 F=$3 RECIPE=$4 P=$5 OBS=$6 O=$OUT/$2 TAG=${2%%_*}
  mkdir -p "$O"; cp "$OUT/start/gate.jsonl" "$O/gate_s00.jsonl"; cp "$OUT/start/gate.jsonl.summary.json" "$O/gate_s00.jsonl.summary.json"
  local R=(--arm fp32 --opt sign --host "$HOST")
  [ "$RECIPE" = ours ] && R=(--arm fp32_renorm --opt normgrad --host "$HOST_RENORM")
  stamp "${NAME}_START gpu$G $RECIPE per_step=$P obs=$OBS"
  ( gpu_env "$G"; cd /tmp && "$PY" "$HERE/bf16_exp.py" stream --source "$SRC" --refdir "$REFS" --ref "$BASE" "${R[@]}" \
      --start "$CK" ${MASTER_ARG[@]+"${MASTER_ARG[@]}"} --ff "$F" --ff-tok "$CK" --ff-tag "$TAG" --ff-per-step "$P" --ff-seed "$SEED" \
      --probe-turn -1 --gate-tasks "$GATE" --gate-py "$PYV" --observe-steps "$OBS" --cand-dir "$OUT/cand_$NAME" --out "$O" > "$O.out" 2>&1 )
  stamp "${NAME}_EXIT $?"; rm -rf "${OUT:?}/cand_$NAME"
}
for g in "${!GPUS[@]}"; do
  ( for j in "${!ARMS[@]}"; do (( j % NG == g )) && run_arm "${GPUS[$g]}" ${ARMS[$j]}; done ) &
done; wait
stamp ALL_DONE
"$PY" "$HERE/compare_ff.py" "$OUT" | tee "$OUT/summary.txt"
