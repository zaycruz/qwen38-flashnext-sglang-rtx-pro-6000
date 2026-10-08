#!/usr/bin/env python3
"""Prefill/decode fairness probe for the GLM sglang engine on asus-trx50.

Measures how a large cold prefill starves small requests, and how it degrades
co-tenant decode. Run it inside a throwaway container with host networking:

    docker run --rm --network host \
      -v /home/zay/sglang-ds4-sm120-pr34717:/bench:ro \
      -v /home/zay/bench-results:/out \
      -v /path/to/this/repo/scripts/inference:/probe:ro \
      python:3.12-slim sh -c "pip -q install aiohttp && python3 /probe/probe-prefill-fairness.py --output /out/fairness-<ts>.json"

The probe sends normal completion requests only. It changes no engine state.
All heavy scenarios use a unique prompt salt, so the radix cache stays cold.

Scenarios (see --scenarios):
  small_cold_alone     Two small requests, cold. Baseline time to first token.
  small_warm_alone     Same small prompt twice. Baseline warm TTFT.
  big_alone_50k        One ~50K-token cold prefill, one output token.
                       TTFT equals the full prefill wall time.
  starved_cold_50k     ~50K cold prefill starts, then a small cold request
                       starts 0.5 s later. Measures TTFT starvation.
  starved_warm_50k     Same, but the small prompt is fully prefix-cached.
  decode_during_flood  A small stream decodes, then a ~50K prefill starts.
                       Measures the stream's delivered rate before, during,
                       and after the prefill.
  starved_cold_10k     starved_cold with a ~10K prefill.
  starved_cold_25k     starved_cold with a ~25K prefill.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

MODEL = "glm-5.3-flash-sglang"
PROM_URL = "http://127.0.0.1:9090"

PROMPT_TEMPLATES = (
    "Debug this failing agent task. Identify the likely root cause, propose the smallest safe fix, and give exact verification commands.",
    "Implement this coding change as an agent. State the files to inspect, the boundary to preserve, the edit, and the focused tests to run.",
    "Review this patch as a senior engineer. Rank correctness, reliability, and scope risks, then state the evidence needed for each finding.",
    "Coordinate this coding workflow. Produce a concise diagnosis, implementation, and verification plan, including rollback conditions.",
)
BLOCK = (
    "Repository context: this service is part of a multi-agent coding system. "
    "A worker receives an issue, inspects a TypeScript service, updates one narrow "
    "module, runs focused tests, and reports evidence. The codebase uses explicit "
    "boundaries, retries only at safe edges, structured JSON logs, and failure "
    "messages that distinguish product defects from environment blockers. Existing "
    "behavior must be preserved outside the requested slice. "
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def small_prompt(index: int, salt: str | None) -> str:
    """The recipe benchmark's ~1761-token prompt, optionally made unique."""
    ctx = (BLOCK * 22).strip()
    head = f"Agent request {index}: {PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)]}\n\n"
    if salt:
        head = f"Salt {salt}: {head}"
    return (
        f"{head}{ctx}\n\n"
        "Issue-specific context: inspect the retry path, preserve cancellation and "
        "single-flight semantics, avoid a framework rewrite, and cite concrete tests."
    )


def big_prompt(target_tokens: int, chars_per_token: float, salt: str) -> str:
    """A cold, unique prompt of about target_tokens tokens."""
    budget = int(target_tokens * chars_per_token)
    parts = [f"Salt {salt}: analyze the system described below."]
    filler_len = 0
    while filler_len < budget:
        parts.append(BLOCK)
        filler_len += len(BLOCK)
    parts.append("Summarize the main risk in one sentence.")
    return "\n".join(parts)


async def stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Run one streaming request and record per-token arrival times."""
    started = time.time()
    first_token: float | None = None
    token_times: list[float] = []
    usage: dict[str, Any] = {}
    status = None
    error = ""
    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as response:
        status = response.status
        if status != 200:
            error = (await response.text())[:500]
        else:
            async for raw in response.content:
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if not any(
                        delta.get(k) not in (None, "", [], {})
                        for k in ("content", "reasoning_content", "reasoning",
                                  "tool_calls")
                    ):
                        continue
                    now = time.time()
                    if first_token is None:
                        first_token = now
                    token_times.append(now)
    ended = time.time()
    completion_tokens = int(usage.get("completion_tokens") or len(token_times))
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached_tokens = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached_tokens = int(details.get("cached_tokens") or 0)
    ok = status == 200 and first_token is not None and completion_tokens >= 1
    gaps_ms = [
        (b - a) * 1000 for a, b in zip(token_times, token_times[1:])
    ]
    decode_s = max(ended - first_token, 1e-9) if first_token else 0.0
    emitted = max(completion_tokens - 1, 0)
    return {
        "success": ok,
        "status": status,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": (first_token - started) * 1000 if first_token else None,
        "e2e_ms": (ended - started) * 1000,
        "tok_s": emitted / decode_s if decode_s > 0 else 0.0,
        "gap_p50_ms": percentile(gaps_ms, 50),
        "gap_max_ms": max(gaps_ms) if gaps_ms else 0.0,
        "n_token_events": len(token_times),
        "token_times": token_times,
        "started_epoch": started,
        "ended_epoch": ended,
    }


def small_body(index: int, salt: str | None, out_tokens: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": small_prompt(index, salt)}],
        "temperature": 0,
        "seed": 70000 + index,
        "max_tokens": out_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def big_body(prompt: str, out_tokens: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": out_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def prom_query(
    session: aiohttp.ClientSession, query: str, at: float | None = None
) -> Any:
    """One Prometheus instant query. Returns the first result value or None."""
    params = [("query", query)]
    if at is not None:
        params.append(("time", f"{at:.0f}"))
    try:
        async with session.get(f"{PROM_URL}/api/v1/query", params=params) as resp:
            data = await resp.json(content_type=None)
        result = (data.get("data") or {}).get("result") or []
        if not result:
            return None
        value = result[0].get("value")
        return None if not value else value[1]
    except Exception:
        return None


async def prom_query_range_max(
    session: aiohttp.ClientSession, query: str, start: float, end: float
) -> Any:
    """Max of a range query. Used for queue-depth gauges."""
    params = [
        ("query", query),
        ("start", f"{start:.0f}"),
        ("end", f"{end:.0f}"),
        ("step", "5s"),
    ]
    try:
        async with session.get(f"{PROM_URL}/api/v1/query_range", params=params) as resp:
            data = await resp.json(content_type=None)
        result = (data.get("data") or {}).get("result") or []
        if not result:
            return None
        values = [float(v[1]) for v in result[0].get("values", []) if v[1] != "NaN"]
        return max(values) if values else None
    except Exception:
        return None


GAUGE_CANDIDATES = [
    "sglang:num_running_reqs",
    "sglang:num_running_requests",
    "sglang:num_queue_reqs",
    "sglang:num_queue_requests",
    "sglang:num_waiting_reqs",
    "sglang:num_waiting_requests",
]


async def engine_side(
    session: aiohttp.ClientSession, start: float, end: float
) -> dict[str, Any]:
    """Engine-side view of one scenario window, from Prometheus."""
    window = max(int(end - start) + 10, 30)
    itl = await prom_query(
        session,
        "histogram_quantile(0.5, sum by (le) "
        f"(rate(sglang:inter_token_latency_seconds_bucket[{window}s])))",
        at=end,
    )
    gen_rate = await prom_query(
        session,
        f"sum(rate(sglang:generation_tokens_total[{window}s]))",
        at=end,
    )
    token_usage = await prom_query(session, "sglang:token_usage", at=end)
    out: dict[str, Any] = {
        "window_s": window,
        "itl_p50_s": itl,
        "generation_tokens_per_s": gen_rate,
        "token_usage_at_end": token_usage,
        "max_queue_depth": None,
    }
    for name in GAUGE_CANDIDATES:
        value = await prom_query_range_max(session, name, start, end)
        if value is not None:
            out["max_queue_depth"] = {"metric": name, "max": value}
            break
    return out


async def wait_idle(session: aiohttp.ClientSession, base_url: str, deadline: float):
    """Wait until the engine reports no running requests."""
    while time.time() < deadline:
        usage = await prom_query(session, "sglang:token_usage")
        if usage is None or float(usage) <= 0.001:
            return
        await asyncio.sleep(2)


class Probe:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.chars_per_token = args.chars_per_token
        self.results: list[dict[str, Any]] = []

    async def run_scenario(
        self,
        session: aiohttp.ClientSession,
        name: str,
        coro_factory,
    ) -> dict[str, Any]:
        base_url = self.args.base_url
        await wait_idle(session, base_url, time.time() + 60)
        start = time.time()
        record = await coro_factory(session)
        end = time.time()
        record["scenario"] = name
        record["window"] = {
            "start_epoch": start,
            "end_epoch": end,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)),
        }
        record["engine_side"] = await engine_side(session, start, end)
        self.results.append(record)
        print(f"[{name}] done: {json.dumps({k: v for k, v in record.items() if k in ('summary',)})}",
              flush=True)
        return record

    # -- scenario builders ---------------------------------------------

    async def small_cold_alone(self, session: aiohttp.ClientSession):
        rows = []
        for i in range(2):
            salt = uuid.uuid4().hex[:8]
            row = await stream_request(
                session,
                self.args.base_url,
                small_body(i, salt, self.args.small_out_tokens, self.args.model),
                self.args.timeout,
            )
            rows.append(row)
            if row["success"]:
                chars = len(small_prompt(i, salt))
                self.chars_per_token = chars / max(row["prompt_tokens"], 1)
        ok = [r for r in rows if r["success"]]
        return {
            "summary": {
                "ttft_ms": [r["ttft_ms"] for r in ok],
                "tok_s": [round(r["tok_s"], 2) for r in ok],
                "chars_per_token_est": round(self.chars_per_token, 3),
            },
            "requests": [strip_tokens(r) for r in rows],
        }

    async def small_warm_alone(self, session: aiohttp.ClientSession):
        salt = uuid.uuid4().hex[:8]
        rows = []
        for _ in range(2):
            row = await stream_request(
                session,
                self.args.base_url,
                small_body(0, salt, self.args.small_out_tokens, self.args.model),
                self.args.timeout,
            )
            rows.append(row)
        return {
            "summary": {
                "ttft_ms_prime": rows[0]["ttft_ms"],
                "ttft_ms_warm": rows[1]["ttft_ms"] if len(rows) > 1 else None,
                "cached_tokens_warm": rows[1]["cached_tokens"] if len(rows) > 1 else None,
            },
            "requests": [strip_tokens(r) for r in rows],
        }

    async def big_alone(self, session: aiohttp.ClientSession, target: int):
        salt = uuid.uuid4().hex[:8]
        prompt = big_prompt(target, self.chars_per_token, salt)
        row = await stream_request(
            session, self.args.base_url, big_body(prompt, 1, self.args.model), self.args.timeout
        )
        actual = row["prompt_tokens"]
        if actual and abs(actual - target) / target > 0.15:
            ratio = len(prompt) / actual
            self.chars_per_token = ratio
        return {
            "summary": {
                "target_tokens": target,
                "prompt_tokens": actual,
                "prefill_wall_ms": row["ttft_ms"],
            },
            "requests": [strip_tokens(row)],
        }

    async def starved(
        self,
        session: aiohttp.ClientSession,
        target: int,
        warm: bool,
    ):
        label = "starved_warm" if warm else "starved_cold"
        big_salt = uuid.uuid4().hex[:8]
        big_prompt_text = big_prompt(target, self.chars_per_token, big_salt)

        warm_salt = uuid.uuid4().hex[:8]
        if warm:
            for _ in range(2):
                await stream_request(
                    session,
                    self.args.base_url,
                    small_body(0, warm_salt, 64, self.args.model),
                    self.args.timeout,
                )

        async def big():
            return await stream_request(
                session,
                self.args.base_url,
                big_body(big_prompt_text, 1, self.args.model),
                self.args.timeout,
            )

        async def small():
            await asyncio.sleep(0.5)
            return await stream_request(
                session,
                self.args.base_url,
                small_body(0, warm_salt if warm else uuid.uuid4().hex[:8],
                           self.args.small_out_tokens, self.args.model),
                self.args.timeout,
            )

        big_row, small_row = await asyncio.gather(big(), small())
        return {
            "summary": {
                "label": label,
                "big_target_tokens": target,
                "big_prompt_tokens": big_row["prompt_tokens"],
                "big_prefill_wall_ms": big_row["ttft_ms"],
                "small_ttft_ms": small_row["ttft_ms"],
                "small_cached_tokens": small_row["cached_tokens"],
                "small_tok_s_after": round(small_row["tok_s"], 2),
            },
            "requests": {
                "big": strip_tokens(big_row),
                "small": strip_tokens(small_row),
            },
        }

    async def decode_during_flood(self, session: aiohttp.ClientSession, target: int):
        big_salt = uuid.uuid4().hex[:8]
        big_prompt_text = big_prompt(target, self.chars_per_token, big_salt)
        small_salt = uuid.uuid4().hex[:8]

        async def small():
            return await stream_request(
                session,
                self.args.base_url,
                small_body(0, small_salt, self.args.flood_stream_tokens, self.args.model),
                self.args.timeout,
            )

        async def big(delayed_start: dict[str, Any]):
            await asyncio.sleep(1.0)
            delayed_start["epoch"] = time.time()
            row = await stream_request(
                session,
                self.args.base_url,
                big_body(big_prompt_text, 1, self.args.model),
                self.args.timeout,
            )
            delayed_start["end_epoch"] = time.time()
            return row

        flood_mark: dict[str, Any] = {}
        small_row, big_row = await asyncio.gather(small(), big(flood_mark))

        times = small_row["token_times"]
        flood_start = flood_mark.get("epoch")
        flood_end = flood_mark.get("end_epoch")

        def rate_in(a: float, b: float) -> float | None:
            inside = [t for t in times if a <= t < b]
            if len(inside) < 2:
                return None
            return (len(inside) - 1) / (inside[-1] - inside[0])

        pre = rate_in(times[0], flood_start) if times and flood_start else None
        during = rate_in(flood_start, flood_end) if flood_start and flood_end else None
        post = rate_in(flood_end, times[-1] + 1e-9) if flood_end and times else None
        during_gaps = [
            (b - a) * 1000
            for a, b in zip(times, times[1:])
            if flood_start <= b < flood_end
        ]
        return {
            "summary": {
                "big_prompt_tokens": big_row["prompt_tokens"],
                "big_prefill_wall_ms": big_row["ttft_ms"],
                "small_tok_s_preflood": round(pre, 2) if pre else None,
                "small_tok_s_during_flood": round(during, 2) if during else None,
                "small_tok_s_postflood": round(post, 2) if post else None,
                "small_gap_p50_ms_during_flood": round(percentile(during_gaps, 50), 2)
                if during_gaps
                else None,
                "small_gap_max_ms_during_flood": round(max(during_gaps), 2)
                if during_gaps
                else None,
            },
            "flood": {
                "start_epoch": flood_start,
                "end_epoch": flood_end,
            },
            "requests": {
                "big": strip_tokens(big_row),
                "small": strip_tokens(small_row),
            },
        }


def strip_tokens(row: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulky token_times list from a request record."""
    out = dict(row)
    out.pop("token_times", None)
    return out


SCENARIOS = {
    "small_cold_alone": lambda p: p.small_cold_alone,
    "small_warm_alone": lambda p: p.small_warm_alone,
    "big_alone_50k": lambda p: (lambda s: p.big_alone(s, 50000)),
    "starved_cold_50k": lambda p: (lambda s: p.starved(s, 50000, warm=False)),
    "starved_warm_50k": lambda p: (lambda s: p.starved(s, 50000, warm=True)),
    "decode_during_flood_50k": lambda p: (
        lambda s: p.decode_during_flood(s, 50000)
    ),
    "starved_cold_25k": lambda p: (lambda s: p.starved(s, 25000, warm=False)),
    "starved_cold_10k": lambda p: (lambda s: p.starved(s, 10000, warm=False)),
}
DEFAULT_SCENARIOS = list(SCENARIOS)


async def get_server_info(session: aiohttp.ClientSession, base_url: str):
    try:
        async with session.get(f"{base_url.rstrip('/')}/get_server_info") as resp:
            return await resp.json(content_type=None)
    except Exception:
        return {}


def pick_server_args(info: dict[str, Any]) -> dict[str, Any]:
    args = info.get("server_args") or info
    if not isinstance(args, dict):
        return {}
    keys = (
        "chunked_prefill_size",
        "max_prefill_tokens",
        "max_running_requests",
        "enable_mixed_chunk",
        "speculative_algorithm",
        "schedule_policy",
        "max_total_num_tokens",
        "page_size",
    )
    return {k: args.get(k) for k in keys if k in args}


async def main_async(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=16)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        info = await get_server_info(session, args.base_url)
        probe = Probe(args)
        picked = pick_server_args(info)
        print(f"engine args: {json.dumps(picked)}", flush=True)
        for name in args.scenarios:
            if name not in SCENARIOS:
                raise SystemExit(f"unknown scenario: {name}")
            await probe.run_scenario(session, name, SCENARIOS[name](probe))
            await asyncio.sleep(args.gap_seconds)

        output = {
            "probe": "prefill-fairness",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": args.base_url,
            "small_out_tokens": args.small_out_tokens,
            "engine_args": picked,
            "scenarios": probe.results,
        }
        text = json.dumps(output, indent=1)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path}", flush=True)
        else:
            print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="space-separated scenario names, in order",
    )
    parser.add_argument("--small-out-tokens", type=int, default=512)
    parser.add_argument(
        "--flood-stream-tokens",
        type=int,
        default=2048,
        help="output tokens for the decode stream in decode_during_flood; long enough to outlast the flood",
    )
    parser.add_argument("--chars-per-token", type=float, default=3.6)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--gap-seconds", type=float, default=5)
    parser.add_argument("--output", default=None, help="write combined JSON here")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
