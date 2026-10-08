# Flash-Next on sglang TP2 - autoresearch loop (2026-10-07)

Goal: decode + prefill comparable to the vLLM TP2 FP8 prod engine
(C1 175 tok/s p50, TPOT 5.7 ms, TTFT 186 ms; C24 36.6/stream, agg 700, TTFT 3.1 s).
Engine: sglang dev-qwen38-next-local g4ccff141, TP2, RadixArk NVFP4, one pinned PLE table.
Bench: benchmark_agentic realistic coding prompts (~1761 in / 512 out), levels 1 and 24.

| # | config delta | boot | C1 p50 tok/s | C1 TPOT | C1 TTFT | C24 p50 | C24 agg | C24 TTFT | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A1 | mtp, decode graphs disabled (eager) | OK | 41.1 | 24.3 ms | 141 ms | 34.6 | 662 | 1180 ms | baseline; decode 4x off target, prefill better than vLLM |
| A2 | mtp, decode graphs FULL, --disable-custom-all-reduce (NCCL AR) | OK, all graphs captured | 223.8 | 4.47 ms | 90 ms | 66.9 | 1164.5 | 2373 ms | WINS vs vLLM on every axis; root cause: intra-node custom AR broke decode graph capture |
| A3 | A2 + --enable-torch-symm-mem | OK, all graphs captured | 226.8 | 4.41 ms | 92 ms | 64.7 | 1157.7 | 2338 ms | tie with A2 (noise); A2 wins by parsimony |
| A4 | nomtp cell, decode graphs full, --disable-custom-all-reduce (KV 3.29M, 64 slots) | OK | 118.5 | 8.44 ms | 86 ms | 58.9 | 1130.8 | 2186 ms | crossover: MTP wins <=C24, no-MTP wins C32 (agg 1604.6, TTFT 926 ms) |
| FINAL | A2 canonical (mtp + full graphs + --disable-custom-all-reduce) full sweep | OK | 211.7 (C1) | 4.72 ms | 89 ms | 67.9 | 1414.3 | 827 ms | WINNER: beats vLLM TP2 FP8 on every metric; flood probe: co-tenant keeps ~91 tok/s (42%) during 50K prefill; warm cache hit 98.8 ms |
