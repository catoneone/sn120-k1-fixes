#!/usr/bin/env bash
# Head x batch 2x2 (7fa0305a / 6b16e298 x e197-e206 / e226-e235) plus the extra 6b16e298 arms. See EXPERIMENT-2x2.md.
# One arm per GPU at a time; arms are spread round-robin over $GPUS and run in order on each GPU.
#
# Required env:
#   PY           training venv python (torch 2.14.0, transformers 5.14.1, flash-linear-attention/fla-core 0.6.0, boto3, pynacl)
#   PYV          vLLM venv python (vllm 0.32.0, math-verify 0.9.0)
#   SRC          signed source tree (bundle a2a03cb3...)
#   DATA         dir with turns/<e>.json and docs/<sha256>.json for e196-e206 and e226-e235 (code/fetch_k1.py)
#   BASE         MiMo base checkpoint (the KL reference)
#   CK_7FA       checkpoint dir of head 7fa0305a
#   CK_6B        checkpoint dir of head 6b16e298
#   OUT          results dir
# Optional env:
#   HOST_RENORM  /k1/replay/host/ files with patch/support_renorm.diff applied (default: downloaded from affine.io and patched here,
#                together with /k1/replay/replay_k1.py, into code/replay/; these are byte-identical to what our runs used)
#   GATE         gate task file (default: data/gate_B.json, 977 tasks, none used in e196-e245)
#   GPUS         default "0 1 2 3 4 5 6 7"
set -uo pipefail
: "${PY:?}" "${PYV:?}" "${SRC:?}" "${DATA:?}" "${BASE:?}" "${CK_7FA:?}" "${CK_6B:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
get() { curl -sSL --fail --retry 3 --max-time 180 -A sn120-replay/1 -o "$2" "https://affine.io/k1/replay/$1"; }
if [ ! -f "$HERE/replay/replay_k1.py" ]; then
  mkdir -p "$HERE/replay/host" && get replay_k1.py "$HERE/replay/replay_k1.py" || { echo 'cannot fetch replay_k1.py'; exit 1; }
fi
if [ -z "${HOST_RENORM:-}" ]; then
  HOST_RENORM=$HERE/replay/host_renorm
  if [ ! -f "$HOST_RENORM/v8211_batched.py" ]; then
    mkdir -p "$HERE/replay/build/host"
    for f in v8211_batched.py v83_fast_kernels.py v83_dp.py; do get "host/$f" "$HERE/replay/build/host/$f" || { echo "cannot fetch $f"; exit 1; }; done
    (cd "$HERE/replay/build" && patch -p1 -s < "$HERE/../patch/support_renorm.diff") || { echo 'patch failed'; exit 1; }
    mv "$HERE/replay/build/host" "$HOST_RENORM" && rm -rf "$HERE/replay/build"
  fi
fi
GATE=${GATE:-$HERE/../data/gate_B.json}; GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); NG=${#GPUS[@]}
REFS=$OUT/refcache; mkdir -p "$OUT" "$REFS"
stamp() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/progress.txt"; }
gpu_env() { export CUDA_VISIBLE_DEVICES=$1 TRITON_CACHE_DIR=$OUT/triton$1 VLLM_CACHE_ROOT=$OUT/vllm_cache$1 PYTHONUNBUFFERED=1 \
                   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; }

# 1. KL reference log-probs for every trained turn (independent of the start weights), split over the GPUs
TURNS=(197 198 199 200 201 202 203 204 205 206 226 227 228 229 230 231 232 233 234 235)
for i in "${!GPUS[@]}"; do
  mine=(); for j in "${!TURNS[@]}"; do (( j % NG == i )) && mine+=("${TURNS[$j]}"); done
  [ ${#mine[@]} -eq 0 ] && continue
  ( gpu_env "${GPUS[$i]}"; cd /tmp && "$PY" "$HERE/bf16_exp.py" refs --source "$SRC" --data "$DATA" --refdir "$REFS" --ref "$BASE" \
      --turns "$(IFS=,; echo "${mine[*]}")" > "$OUT/refs_${GPUS[$i]}.out" 2>&1 || stamp REFS_FAIL "${mine[*]}" ) &
done; wait
stamp REFS_DONE "$(ls "$REFS"/*.pt | wc -l) of ${#TURNS[@]}"

# 2. Start evals of both heads on the gate set (each arm's deltas are paired against these)
start_eval() { ( gpu_env "$1"; cd "$HERE" && "$PYV" gate_eval.py --ckpt "$2" --tasks "$GATE" --out "$OUT/start_$3/gate.jsonl" \
                 > "$OUT/start_$3.out" 2>&1 ) && stamp START_EVAL_$3 "$(cat "$OUT/start_$3/gate.jsonl.summary.json")" || stamp START_EVAL_FAIL_$3; }
if (( NG >= 2 )); then start_eval "${GPUS[0]}" "$CK_7FA" 7fa & start_eval "${GPUS[1]}" "$CK_6B" 6b & wait
else start_eval "${GPUS[0]}" "$CK_7FA" 7fa; start_eval "${GPUS[0]}" "$CK_6B" 6b; fi

# 3. Arms: NAME START TURNS STEP_MULT OBS ("gated" = gate every step, rollback of bf16 weights and fp32 master on reject)
#    fp32 master + support renorm + gradient-direction step (normgrad) in all arms; --step-mult scales each tensor's step.
ARMS=(
  "A7fa_e226_1x   7fa 226-235          1    1,3,6,10"
  "B6b_e197_1x    6b  197-206          1    1,3,6,10"
  "B6b_e226_1x    6b  226-235          1    1,3,6,10"
  "B6b_e226_05x   6b  226-235          0.5  1,3,6,10"
  "B6b_e226_025x  6b  226-235          0.25 1,3,6,10"
  "A7fa_e197_1x   7fa 197-206          1    1,3,6,10"
  "A7fa_e226_05x  7fa 226-235          0.5  1,3,6,10"
  "G6b_e226_1x    6b  226-235          1    gated"
  "S6b_skip_025x  6b  226-228,232-235  0.25 3,4,7"
  "A7fa_e197_05x  7fa 197-206          0.5  1,3,6,10"
  "B6b_e197_05x   6b  197-206          0.5  1,3,6,10"
)
run_arm() {  # GPU NAME HEAD TURNS MULT OBS
  local G=$1 NAME=$2 H=$3 TURNS=$4 MULT=$5 OBS=$6 START O=$OUT/$2
  [ "$H" = 7fa ] && START=$CK_7FA || START=$CK_6B
  [ -f "$OUT/start_$H/gate.jsonl.summary.json" ] || { stamp "${NAME}_NO_START_EVAL"; return 1; }
  mkdir -p "$O"; cp "$OUT/start_$H/gate.jsonl" "$O/gate_s00.jsonl"; cp "$OUT/start_$H/gate.jsonl.summary.json" "$O/gate_s00.jsonl.summary.json"
  local EXTRA="--observe-steps $OBS"; [ "$OBS" = gated ] && EXTRA=""
  stamp "${NAME}_START gpu$G $H $TURNS x$MULT $OBS"
  ( gpu_env "$G"; cd /tmp && "$PY" "$HERE/bf16_exp.py" stream --source "$SRC" --data "$DATA" --refdir "$REFS" --ref "$BASE" \
      --host "$HOST_RENORM" --arm fp32_renorm --opt normgrad --step-mult "$MULT" --start "$START" --turns "$TURNS" --probe-every 100 \
      --gate-tasks "$GATE" --gate-py "$PYV" $EXTRA --cand-dir "$OUT/cand_$NAME" --out "$O" > "$O.out" 2>&1 )
  stamp "${NAME}_EXIT $?"; rm -rf "${OUT:?}/cand_$NAME"
}
for i in "${!GPUS[@]}"; do
  ( for j in "${!ARMS[@]}"; do (( j % NG == i )) && run_arm "${GPUS[$i]}" ${ARMS[$j]}; done ) &
done; wait
stamp ALL_DONE
grep -E 'OBSERVE|GATE [0-9]' "$OUT"/*/log.txt > "$OUT/summary.txt" 2>/dev/null
