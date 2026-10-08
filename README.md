# Qwen3.8-Flash-Next on SGLang — 2× RTX PRO 6000 (TP2) recipe and benchmarks

A working, measured recipe for serving **Qwen3.8-Flash-Next** (125B MoE, ~6B active) on two
RTX PRO 6000 Blackwell cards (96 GB each, SM120) with **SGLang**, **NVFP4** weights, and
**NEXTN MTP speculative decoding** — plus the one bug that was silently costing us 5× on decode.

Measured on 2026-10-07/08 with sglang `dev-qwen38-next-local` (build `g4ccff141`) and the
`RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint. The same box also runs a tuned
vLLM TP2 FP8 production engine; both are measured below with identical clients and prompts.

## The bug worth knowing about

**SGLang's intra-node custom allreduce breaks decode CUDA graph capture at TP2** on this
build. Symptom chain:

1. Decode graphs fail to capture with `Capture cuda graph failed: invalid argument`.
2. SGLang masks the failure and falls back to **eager decode** — no crash, just a warning
   in the logs.
3. Decode runs ~5× slower than it should (41 tok/s instead of 212 at C1), and nothing
   tells you why unless you read the boot log.

The fix is one flag:

```
--disable-custom-all-reduce
```

This switches intra-node TP2 communication to NCCL allreduce. With it, all decode graphs
capture and single-stream decode goes from 41 → 212 tok/s. The prefill path was unaffected;
this is purely a decode-graph-capture interaction.

Minimal repro on this hardware/build: default flags → capture fails → eager fallback;
`--disable-custom-all-reduce` → full graph capture. Both behaviors are visible in the boot
log (`Capture cuda graph` lines).

## Hardware and memory shape

| Component | Value |
|---|---|
| GPUs | 2× RTX PRO 6000 (Blackwell, SM120), 96 GB each, TP2 |
| Weights | NVFP4, ~39 GB per card (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) |
| KV / Mamba pool | 2.69M tokens (mtp cell) / 3.29M (nomtp cell) |
| Host RAM pinned | ~48 GB for the PLE n-gram table (one table only — see warning) |
| Host RAM floor | 70 GB free required by the launcher preflight |

> **Host-RAM warning:** the PLE pinned n-gram table costs ~48 GB of *host* RAM. Two
> pinned tables (e.g. two TP1 replicas) on a 123 GB host caused a kernel OOM that killed
> `systemd --user` and the network stack — full power cycle required. Run **one engine,
> one pinned table**, and keep a memory preflight in front of every engine start.

## The winning config (mtp cell)

```bash
python3 -m sglang.launch_server \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen3.8-flash-next \
  --tp 2 \
  --chunked-prefill-size 8192 \
  --moe-runner-backend flashinfer_cutlass \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --ple-offload-embedding --ple-offload-backend pinned \
  --reasoning-parser auto --tool-call-parser qwen3_coder \
  --mem-fraction-static 0.90 \
  --max-mamba-cache-size 96 \
  --max-running-requests 32 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce
```

Notes on the non-obvious flags:

- `--tool-call-parser qwen3_coder` — required for agent clients. Without it the model's
  XML tool calls come back as plain text and the client sees no `tool_calls`.

- `--moe-runner-backend flashinfer_cutlass` — required for NVFP4 on SM120; the
  auto-default (`flashinfer_trtllm`) is rejected by this model.
- `--cuda-graph-backend-decode full` — full decode graphs are the whole win; without them
  (eager fallback) decode is 5× slower.
- `--cuda-graph-backend-prefill disabled` — prefill graph capture fails on this build
  ("invalid argument"); prefill runs fine eager, so it is disabled rather than masked.
- `--speculative-algorithm NEXTN` — the checkpoint ships its own MTP head. Measured
  accept length: **2.45**. It wins at every concurrency up to C24; past ~C32 the no-MTP
  cell wins (see crossover below).

The `nomtp` cell (no speculative decoding, `--max-running-requests 64`,
`--max-mamba-cache-size 192`) is in the launcher for high-concurrency serving.

## Benchmarks

Method: realistic agentic-coding prompts (~1761 tokens in, 512 out), closed-loop
concurrency levels, one warmup then L simultaneous streams; p50 per-stream throughput,
aggregate output tok/s, TTFT and TPOT from the streaming client. Same client, same
prompts, same box, same night for both engines.

### Concurrency sweep: sglang NVFP4 TP2 (winner) vs vLLM FP8 TP2 (tuned prod engine)

The vLLM engine is *not* a strawman — it runs a considered production config: custom
allreduce disabled, CUDA graphs captured to batch size 64, `max-num-seqs 64`, MTP
speculative decoding (3 tokens), prefix caching on.

| C | sglang p50 tok/s | sglang agg | sglang TTFT p50 | sglang TPOT | vLLM p50 tok/s | vLLM agg | vLLM TTFT p50 | vLLM TPOT |
|---|---|---|---|---|---|---|---|---|
| 1 | **211.7** | 205 | **89 ms** | 4.72 ms | 175.0 | 165 | 186 ms | 5.71 ms |
| 4 | **165.1** | **545** | **369 ms** | 6.06 ms | 108.1 | 296 | 2066 ms | 9.25 ms |
| 8 | **121.9** | **830** | **599 ms** | 8.20 ms | 84.1 | 509 | 1898 ms | 11.90 ms |
| 16 | **87.6** | **1147** | **1096 ms** | 11.42 ms | 57.9 | 720 | 2140 ms | 17.27 ms |
| 24 | **67.9** | **1414** | **827 ms** | 14.72 ms | 36.6 | 700 | 3091 ms | 27.36 ms |
| 32 | 55.3 | 1297 | 1005 ms | 18.09 ms | — | — | — | — |

### Speculative decoding crossover

| Cell | C24 aggregate | C32 aggregate |
|---|---|---|
| MTP (spec decoding) | **1414** | 1297 |
| no MTP (64 slots) | 1131 | **1605** |

MTP wins through C24. No-MTP wins from ~C32 up. If your target concurrency is ≤24, use the
MTP cell; if you push past ~32, switch.

### Prefill fairness probe

What happens to small interactive requests when a giant prefill lands, or while it is
running (cold cache, ~49K-token prefill):

| Probe | sglang NVFP4 | vLLM FP8 |
|---|---|---|
| 50K prefill alone | **3.83 s** | 4.31 s |
| Small req TTFT queued behind 50K prefill | **3.30 s** | 3.82 s |
| Small req TTFT queued behind 25K prefill | **1.46 s** | 1.73 s |
| Small req TTFT queued behind 10K prefill | **369 ms** | 523 ms |
| Co-tenant decode during 50K flood | **37 tok/s (42% of its 88 tok/s baseline)** | 2.9 tok/s (4% of its 67 baseline) |
| Co-tenant after flood | 88 tok/s (full recovery) | 67 tok/s |
| Warm radix-cache TTFT | **98.8 ms** (vs 153.6 cold) | 198.3 ms (no warm benefit) |

The headline: on sglang a co-tenant stream keeps decoding at ~42% of its normal speed
*while* a 49K-token prefill is in flight, and the warm radix cache cuts small-request TTFT
by a third. On the vLLM FP8 config the same co-tenant stalls to ~4%.

## Honest caveats

- **The checkpoints differ.** sglang serves the 4-bit NVFP4 export; vLLM serves the 8-bit
  FP8 export. On 96 GB cards that is the structural difference: FP8 weights eat ~87 GB per
  card and leave almost no KV headroom, which is most of why the vLLM C24 numbers collapse.
  Tuning cannot recover that memory. So this comparison proves "best Flash-Next serving
  config on this hardware beats a tuned vLLM FP8 config on the same hardware" — it does
  *not* prove sglang-the-engine beats vLLM-the-engine on identical weights.
- **NVFP4 quality** is validated in the sglang cookbook (full GSM8K pass at ~97%), and
  4-bit retention is generally good in third-party studies, but there is no coding-specific
  eval for this exact checkpoint here. Gate on an eval before production use.
- **TP2 on RTX PRO 6000 is expected-works, not cookbook-verified** — the cookbook's
  verified cell for this model is TP1 on one card. The numbers above are that verification.
- Dev-build sglang (`dev-qwen38-next-local`); behavior may differ on releases.

## Repo contents

| Path | What it is |
|---|---|
| `scripts/launch-qwen-flashnext-sglang.sh` | The launcher: mtp/nomtp cells, memory preflight, health-wait, log capture. `EXTRA_FLAGS` env appends overrides for experimentation. |
| `scripts/bench-concurrency-sweep.sh` | Concurrency sweep against any OpenAI-compatible endpoint (per-level JSON, GPU samples, engine metrics). |
| `scripts/probe-prefill-fairness.py` | The fairness probe: cold/warm TTFT, starved TTFT at 10K/25K/50K, co-tenant decode during a flood. |
| `docs/autoresearch-ledger.md` | The full A1→A4 attempt ledger (eager baseline, graph-capture fix, symm-mem tie, MTP crossover). |
| `data/*.json` | Raw results: final sweep + fairness probe for sglang, sweep + fairness for the vLLM comparison run. |

## Running it

The launcher expects: the `lmsysorg/sglang:dev-qwen38-next-local` image, the NVFP4 model
in the HF cache, ~70 GB free host RAM, and both GPUs free. Adjust the paths/volumes at the
top of the script for your host.

```bash
./scripts/launch-qwen-flashnext-sglang.sh mtp /tmp/flashnext-run   # serve on :8200
./scripts/bench-concurrency-sweep.sh --label flash \
  --base-url http://127.0.0.1:8200 --model qwen3.8-flash-next
```

To run it as a service, wrap the launcher in a systemd unit (this is a rootless user-unit
pattern; the port and volumes are inside the script):

```ini
[Unit]
Description=SGLang Qwen3.8-Flash-Next TP2 (NVFP4, MTP)
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/home/you/bin/launch-qwen-flashnext-sglang.sh mtp /home/you/logs/flashnext
ExecStop=docker rm -f sglang-flashnext-tp2
TimeoutStartSec=1200

[Install]
WantedBy=default.target
```

## Credits

Benchmarked on a dual RTX PRO 6000 workstation. Raw JSON for every number above is in
`data/`. The autoresearch loop that found the root cause is documented in
`docs/autoresearch-ledger.md`.
