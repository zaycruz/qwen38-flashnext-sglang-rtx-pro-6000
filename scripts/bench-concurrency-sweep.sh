#!/usr/bin/env bash
# Concurrency sweep benchmark for an OpenAI-compatible engine on asus-trx50.
#
# Runs the recipe benchmark (benchmark_agentic.py, aiohttp, realistic agentic
# coding prompts ~1761 tokens, 512 output tokens) once per concurrency level,
# samples GPU state and engine /metrics, and writes one JSON per level plus a
# combined summary to a timestamped results directory.
#
# Run it from any machine that can reach asus-trx50 over SSH:
#   scripts/inference/bench-concurrency-sweep.sh --label NAME \
#     --base-url http://127.0.0.1:30001 --model qwen3.8-flash-next
#
# The engine configuration never changes. The script sends normal completion
# requests only. It reads /metrics when the endpoint serves it; a missing
# /metrics is not an error.
#
# Per level L it runs: warmup (1x128), single (N sequential), concurrent
# (L simultaneous streams, closed loop within the phase). The concurrent
# phase is the measurement: p50 per-stream tok/s, aggregate tok/s, TTFT.
#
# Output layout (under --results-dir, default ~/bench-results/<label>-<ts>/):
#   c{L}.json               full benchmark output for level L
#   c{L}.metrics-before.txt engine /metrics snapshot (if served)
#   c{L}.metrics-after.txt  engine /metrics snapshot (if served)
#   gpu-samples.csv         nvidia-smi samples, one row per 5 s
#   sweep-summary.json      per-level summary rows
#   meta.txt                engine URL, model, levels, flags, start/end times

set -euo pipefail

RECIPE_DIR="${RECIPE_DIR:-/home/zay/sglang-ds4-sm120-pr34717}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/zay/bench-results}"

# Defaults use ":-" so a caller can pass values through the environment.
# The SSH re-exec path below relies on this.
LABEL="${LABEL:-sweep}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
MODEL="${MODEL:-glm-5.3-flash-sglang}"
LEVELS="${LEVELS:-1 4 8 16 24}"
OUTPUT_TOKENS=512
SINGLE_REQUESTS=2

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --single-requests) SINGLE_REQUESTS="$2"; shift 2 ;;
    --results-dir) RESULTS_ROOT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Values travel through an SSH command string. Accept only safe characters.
check_label() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "label must match [A-Za-z0-9._-]" >&2; exit 2; }
}
check_number() {
  [[ "$1" =~ ^[0-9]+$ ]] || { echo "$2 must be a number" >&2; exit 2; }
}
check_path() {
  [[ "$1" =~ ^[/A-Za-z0-9._-]+$ ]] || { echo "$2 has unsafe characters" >&2; exit 2; }
}
check_url() {
  [[ "$1" =~ ^https?://[A-Za-z0-9._:/-]+$ ]] || { echo "$2 has unsafe characters" >&2; exit 2; }
}
check_label "$LABEL"
check_url "$BASE_URL"
check_path "$RECIPE_DIR" "$RECIPE_DIR"
check_path "$RESULTS_ROOT" "$RESULTS_ROOT"
check_number "$OUTPUT_TOKENS" "--output-tokens"
check_number "$SINGLE_REQUESTS" "--single-requests"
for level in $LEVELS; do
  check_number "$level" "--levels"
done

if [ "$(hostname -s)" != "asus-trx50" ]; then
  exec ssh -o BatchMode=yes zay@asus-trx50 \
    "RECIPE_DIR='$RECIPE_DIR' RESULTS_ROOT='$RESULTS_ROOT' \
     LABEL='$LABEL' BASE_URL='$BASE_URL' MODEL='$MODEL' LEVELS='$LEVELS' \
     OUTPUT_TOKENS=$OUTPUT_TOKENS SINGLE_REQUESTS=$SINGLE_REQUESTS \
     bash -s" < "$0"
fi

# ---- everything below runs on asus-trx50 ----

fail() { echo "bench-concurrency-sweep: $*" >&2; exit 1; }

command -v curl >/dev/null || fail "curl is missing"
[ -f "$RECIPE_DIR/benchmark_agentic.py" ] || fail "benchmark tool not found: $RECIPE_DIR"
curl -fsS -o /dev/null --max-time 10 "$BASE_URL/health" || fail "engine health check failed on $BASE_URL"

TS="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="$RESULTS_ROOT/$LABEL-$TS"
mkdir -p "$OUT_DIR"

# Some engines stream reasoning tokens as delta.reasoning (for example the
# vLLM qwen3 parser on Qwen3.8-Flash-Next). The recipe client only knows
# delta.reasoning_content and would count zero output tokens on those
# engines. Patch a copy of the client; never edit the recipe tool itself.
CLIENT="$OUT_DIR/benchmark_agentic.sweep.py"
sed 's/for key in ("content", "reasoning_content", "tool_calls")/for key in ("content", "reasoning_content", "reasoning", "tool_calls")/' \
  "$RECIPE_DIR/benchmark_agentic.py" > "$CLIENT"
grep -q '"reasoning", "tool_calls"' "$CLIENT" \
  || fail "client patch did not apply; recipe tool text changed"

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_url=$BASE_URL"
  echo "model=$MODEL"
  echo "levels=$LEVELS"
  echo "output_tokens=$OUTPUT_TOKENS"
  echo "single_requests=$SINGLE_REQUESTS"
  echo "recipe_dir=$RECIPE_DIR"
  echo "client=patched copy, delta keys include reasoning"
} > "$OUT_DIR/meta.txt"

# GPU sampler, one row every 5 s.
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -l 5 > "$OUT_DIR/gpu-samples.csv" &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" >/dev/null 2>&1 || true' EXIT

snap_metrics() {
  local level="$1" phase="$2"
  curl -s --max-time 5 "$BASE_URL/metrics" \
    > "$OUT_DIR/c${level}.metrics-${phase}.txt" 2>/dev/null || true
}

for level in $LEVELS; do
  echo "== level $level =="
  snap_metrics "$level" before
  docker run --rm --network host \
    -v "$RECIPE_DIR":/bench:ro -v "$OUT_DIR":/out \
    python:3.12-slim sh -c "pip -q install aiohttp 2>/dev/null && \
    python3 /out/$(basename "$CLIENT") \
    --base-url $BASE_URL --model $MODEL \
    --single-requests $SINGLE_REQUESTS --concurrent-streams $level \
    --output-tokens $OUTPUT_TOKENS --output /out/c${level}.json" \
    > "$OUT_DIR/c${level}.stdout.txt" 2>&1 \
    || { cat "$OUT_DIR/c${level}.stdout.txt"; fail "benchmark failed at level $level"; }
  snap_metrics "$level" after
done

kill "$SAMPLER_PID" >/dev/null 2>&1 || true
trap - EXIT
echo "ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT_DIR/meta.txt"

OUT_DIR="$OUT_DIR" /usr/bin/python3 - <<'PY'
import glob, json, os, statistics

out_dir = os.environ["OUT_DIR"]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "c*.json"))):
    with open(path) as fh:
        bench = json.load(fh)
    concurrent = bench.get("concurrent") or {}
    if not concurrent.get("success_rate") == 1.0:
        continue
    detail = concurrent.get("requests_detail") or []
    tpots = [float(r["tpot_ms"]) for r in detail if r.get("success")]
    rows.append({
        "level": concurrent.get("concurrency"),
        "requests": concurrent.get("requests"),
        "p50_per_stream_tok_s": concurrent.get("p50_per_stream_tok_s"),
        "p05_per_stream_tok_s": concurrent.get("p05_per_stream_tok_s"),
        "aggregate_output_tok_s": concurrent.get("aggregate_output_tok_s"),
        "p50_ttft_ms": concurrent.get("p50_ttft_ms"),
        "p95_ttft_ms": concurrent.get("p95_ttft_ms"),
        "p50_tpot_ms": statistics.median(tpots) if tpots else None,
        "wall_seconds": concurrent.get("wall_seconds"),
        "file": os.path.basename(path),
    })

rows.sort(key=lambda r: r["level"] or 0)
summary_path = os.path.join(out_dir, "sweep-summary.json")
with open(summary_path, "w") as fh:
    json.dump(rows, fh, indent=1)

print(f"{'C':>4} {'reqs':>5} {'p50 tok/s':>10} {'agg tok/s':>10} "
      f"{'TTFT p50':>10} {'TTFT p95':>10} {'TPOT p50':>9}")
for r in rows:
    print(f"{r['level']:>4} {r['requests']:>5} "
          f"{r['p50_per_stream_tok_s']:>10.1f} "
          f"{r['aggregate_output_tok_s']:>10.1f} "
          f"{r['p50_ttft_ms']:>10.1f} {r['p95_ttft_ms']:>10.1f} "
          f"{r['p50_tpot_ms']:>9.2f}")
print(f"wrote {summary_path}")
PY
