#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next (RadixArk NVFP4) on asus-trx50 as ONE sglang
# engine across both RTX PRO 6000 cards (TP2), following the sglang cookbook
# Qwen3.8-Flash-Next page (docs.sglang.io).
#
# Run IN the host: launch-qwen-flashnext-sglang.sh {mtp|nomtp} <run-dir>
#   mtp    balanced cell: NEXTN speculative decoding (the checkpoint MTP
#          head, 3 steps, 4 draft tokens), 32 running requests.
#   nomtp  throughput cell: no speculative decoding, 64 running requests.
#
# Memory shape: ONE engine, ONE ~48 GiB n-gram table in pinned host RAM,
# ~39 GiB of weights per card. This mirrors how the GLM engine runs on this
# box (one TP2 engine, one port, one unit). Do not run a second pinned table
# on this host: two pinned tables OOMed the 123 GiB host on 2026-10-07 and
# the kernel killed systemd --user and the network stack (power cycle
# needed; see scripts/oom-guard/).
#
# Note: the cookbook's verified RTX PRO 6000 cell is TP1 (one card). TP2 on
# two PRO 6000s is expected-works, not cookbook-verified; the benchmark
# window is the verification.
#
# Needs: lmsysorg/sglang:dev-qwen38-next-local image, HF cache with
# RadixArk/Qwen3.8-Flash-Next-NVFP4, about 70 GB of free host RAM, and the
# GLM engine stopped (it holds both cards).

set -euo pipefail

MODE=${1:?mode required: mtp or nomtp}
RUN_DIR=${2:?run dir required}
# The autoresearch loop overrides these through the environment. Defaults
# are the working mtp/nomtp cells.
IMAGE=${IMAGE:-lmsysorg/sglang:dev-qwen38-next-local}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
SERVED=qwen3.8-flash-next
MIN_AVAIL_MB=70000
PORT=8200
mkdir -p "$RUN_DIR"

# Preflight: refuse to start without memory headroom. The 2026-10-07 outage
# started exactly here: two pinned tables on a 123 GiB host.
AVAIL_MB=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
if [ "$AVAIL_MB" -lt "$MIN_AVAIL_MB" ]; then
  echo "REFUSING TO START: MemAvailable ${AVAIL_MB} MB < required ${MIN_AVAIL_MB} MB." >&2
  echo "Stop other GPU containers first. One pinned n-gram table (about 48 GB)" >&2
  echo "plus runtime and kernel headroom must fit. See scripts/oom-guard/README.md." >&2
  exit 4
fi

case "$MODE" in
  mtp)
    # The 2026-10-07 autoresearch winner (see ~/bench-results/
    # flashnext-sglang-autoresearch/RESULTS.md on the host):
    # --disable-custom-all-reduce is REQUIRED. The intra-node custom
    # allreduce breaks decode CUDA graph capture at TP2 ("Capture cuda
    # graph failed: invalid argument"); with NCCL allreduce all graphs
    # capture and decode is 5x faster than eager (211 vs 41 tok/s at C1).
    CELL_FLAGS="--mem-fraction-static 0.90 \
      --max-mamba-cache-size 96 --max-running-requests 32 \
      --speculative-algorithm NEXTN --speculative-num-steps 3 \
      --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
      --cuda-graph-backend-decode full \
      --cuda-graph-backend-prefill disabled \
      --disable-custom-all-reduce"
    ;;
  nomtp)
    # Throughput cell. MTP wins through C24 (1414 vs 1131 aggregate);
    # no-MTP wins from ~C32 up (1604 vs 1297 aggregate at C32).
    CELL_FLAGS="--mem-fraction-static 0.90 \
      --max-mamba-cache-size 192 --max-running-requests 64 \
      --cuda-graph-backend-decode full \
      --cuda-graph-backend-prefill disabled \
      --disable-custom-all-reduce"
    ;;
  *)
    echo "mode must be mtp or nomtp" >&2
    exit 2
    ;;
esac

# EXTRA_FLAGS is appended last; argparse takes the last occurrence, so the
# autoresearch loop can override anything above without editing this file.
if [ -n "$EXTRA_FLAGS" ]; then
  CELL_FLAGS="$CELL_FLAGS $EXTRA_FLAGS"
fi

COMMON_FLAGS="--model-path $MODEL --served-model-name $SERVED \
  --host 0.0.0.0 --port 30000 --tp 2 \
  --chunked-prefill-size 8192 \
  --moe-runner-backend flashinfer_cutlass \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --ple-offload-embedding --ple-offload-backend pinned \
  --reasoning-parser auto"

docker rm -f sglang-flashnext-tp2 >/dev/null 2>&1 || true

docker run -d --name sglang-flashnext-tp2 \
  --gpus '"device=0,1"' --ipc=host --shm-size=32g \
  -p "127.0.0.1:${PORT}:30000" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/zay/.cache/huggingface:/root/.cache/huggingface:rw \
  -v /home/zay/.cache/sglang-flashnext:/root/.cache/sglang:rw \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /bin/bash "$IMAGE" -lc \
  "exec python3 -m sglang.launch_server $COMMON_FLAGS $CELL_FLAGS"

ready=false
# 240 x 5 s = 20 min. Model load (78 GiB), n-gram table fill, cuda graphs.
for _ in $(seq 1 240); do
  if curl --fail --silent --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if [[ $(docker inspect -f '{{.State.Running}}' sglang-flashnext-tp2 2>/dev/null || true) != true ]]; then
    docker logs sglang-flashnext-tp2 > "$RUN_DIR/sglang-flashnext-tp2.log" 2>&1 || true
    echo "engine exited during startup; log saved to $RUN_DIR/sglang-flashnext-tp2.log" >&2
    exit 2
  fi
  sleep 5
done
if [[ $ready != true ]]; then
  docker logs sglang-flashnext-tp2 > "$RUN_DIR/sglang-flashnext-tp2.log" 2>&1 || true
  echo "engine timed out during startup; log saved to $RUN_DIR/sglang-flashnext-tp2.log" >&2
  exit 3
fi

docker logs sglang-flashnext-tp2 > "$RUN_DIR/sglang-flashnext-tp2.log" 2>&1
grep -E "max_running_requests|max_total_num_tokens|mamba_cache|PLE|ple_|quantization=|Load weight" \
  "$RUN_DIR/sglang-flashnext-tp2.log" | tail -6 || true

echo "sglang Flash-Next TP2 engine ready on 127.0.0.1:${PORT} (mode=$MODE)"
