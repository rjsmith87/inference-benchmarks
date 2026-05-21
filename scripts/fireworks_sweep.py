"""Run the 25 gold questions against every available Fireworks serverless LLM.

This is the Fireworks counterpart to scripts/baseten_sweep.py and uses the
*identical* protocol so the two sweeps are an apples-to-apples head-to-head in
the dashboard:

  - full 25-question gold set (10 dev + 15 synthetic), not the 10-Q dev subset
  - streaming call with time-to-first-token capture
  - one-shot, no repair loop, no rate-limit retry (cross-model differences stay
    visible rather than being masked by retries)
  - JSON mode, temperature 0, max_tokens 1024
  - same compact+trim schema injection as the shipped agent
  - scoring via src.evals.rows_match (the tolerant matcher)

Output JSON is structurally identical to data/baseten_sweep.json so the
dashboard's _normalizeReport() consumes both with one code path.

Usage:
    PYTHONPATH=. python scripts/fireworks_sweep.py
    PYTHONPATH=. python scripts/fireworks_sweep.py --models accounts/fireworks/models/kimi-k2p6 ...
    PYTHONPATH=. python scripts/fireworks_sweep.py --out data/fireworks_sweep_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Manually load .env (avoids dotenv heredoc-frame quirk).
for ln in Path(".env").read_text().splitlines():
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k, v.strip())

from openai import APIError, OpenAI, RateLimitError

from src.agent import (
    FIREWORKS_BASE_URL, SYSTEM_PROMPT,
    _format_schema_compact, _trim_schema_for_question,
)
from src.evals import rows_match
from src.utils import load_db, query_db

# Pricing per 1M tokens (USD). Only Kimi K2.6 is on the published Fireworks
# pricing page; the rest are not broken out for these model versions, so we
# leave them None — the dashboard proxy-costs them and flags verified=false.
PRICING: dict[str, dict[str, float] | None] = {
    "accounts/fireworks/models/kimi-k2p6":       {"input": 0.95, "output": 4.00},
    "accounts/fireworks/models/kimi-k2p5":       None,
    "accounts/fireworks/models/deepseek-v4-pro": None,
    "accounts/fireworks/models/glm-5p1":         None,
    "accounts/fireworks/models/minimax-m2p7":    None,
}

# Models worth recording but not benchmarkable on this key — a single probe
# call confirms the status so the data file documents *why* they are absent
# from the head-to-head rather than silently dropping them.
PROBE_ONLY = [
    "accounts/fireworks/models/glm-5",
    "accounts/fireworks/models/qwen3-8b",
]

# qwen3-8b is a reasoning model; disable reasoning so it behaves like the
# others if it ever becomes reachable again.
REASONING_OFF = {"accounts/fireworks/models/qwen3-8b"}

# Pacing — Fireworks shared serverless is burst-y and rate-limits hard.
# baseten_sweep.py uses 1.5s because Baseten's tier is permissive; at 1.5s
# Fireworks 429'd 8-14 of 25 calls on 4 of 5 models. 5s clears the 429 storm
# so accuracy reflects the model. The *scored* protocol (streaming, one-shot,
# no repair, no retry, JSON mode, temp 0, scoring) is identical to Baseten's.
INTER_QUERY_SLEEP_S = 5.0
INTER_MODEL_SLEEP_S = 5.0


@dataclass
class CallTrace:
    qid: str
    question: str
    tier: int | None
    raw_content: str
    json_parsed_ok: bool
    sql: str | None
    rows: list[dict] | None
    correct: bool
    ttft_ms: float | None       # time to first non-empty chunk
    latency_ms: float            # total wall time (stream open → final chunk)
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: str | None


@dataclass
class ModelReport:
    model: str
    n: int
    correct: int
    accuracy: float
    json_ok_count: int
    json_ok_rate: float
    median_ttft_ms: float | None
    median_latency_ms: float
    p90_latency_ms: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    cost_per_query_usd: float | None
    pricing: dict[str, float] | None
    fatal_error: str | None
    notes: str
    traces: list[CallTrace] = field(default_factory=list)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def run_one_call(client: OpenAI, model: str, q: dict, conn) -> CallTrace:
    """One streaming call to Fireworks. TTFT + total + tokens + executed SQL."""
    question = q["question"]
    schema = _trim_schema_for_question(
        conn, question, _format_schema_compact(conn), compact=True
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": question},
    ]
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
        response_format={"type": "json_object"},
    )
    if model in REASONING_OFF:
        kwargs["extra_body"] = {"reasoning_effort": "none"}

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    content_parts: list[str] = []
    usage = None

    try:
        stream = client.with_options(timeout=60.0).chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                got_payload = False
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                    got_payload = True
                # Some models emit reasoning_content before content; count it for TTFT
                # since it's the "model started responding" signal users perceive.
                if getattr(delta, "reasoning_content", None):
                    got_payload = True
                if got_payload and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
            if getattr(chunk, "usage", None):
                usage = chunk.usage
    except (APIError, RateLimitError, Exception) as e:
        return CallTrace(
            qid=q["id"], question=question, tier=q.get("tier"),
            raw_content="", json_parsed_ok=False, sql=None, rows=None, correct=False,
            ttft_ms=ttft_ms,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )

    total_ms = (time.perf_counter() - t0) * 1000
    content = "".join(content_parts)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0

    rates = PRICING.get(model)
    cost = (pt * rates["input"] + ct * rates["output"]) / 1_000_000 if rates else 0.0

    json_ok = False
    sql: str | None = None
    try:
        parsed = json.loads(content)
        json_ok = isinstance(parsed, dict) and "sql" in parsed
        if json_ok:
            sql = (parsed.get("sql") or "").strip().rstrip(";").strip()
    except (json.JSONDecodeError, TypeError):
        pass

    rows: list[dict] | None = None
    err: str | None = None
    if not sql:
        err = "no SQL extractable" if not json_ok else "model returned empty SQL"
    else:
        try:
            rows = query_db(conn, sql, return_as_df=False)
        except Exception as e:
            err = f"sqlite: {e}"

    correct = False
    if rows is not None and not err and q.get("expected_result") is not None:
        correct = rows_match(rows, q["expected_result"])

    return CallTrace(
        qid=q["id"], question=question, tier=q.get("tier"),
        raw_content=content[:500], json_parsed_ok=json_ok,
        sql=sql, rows=rows, correct=correct,
        ttft_ms=ttft_ms, latency_ms=total_ms,
        prompt_tokens=pt, completion_tokens=ct, cost_usd=cost,
        error=err,
    )


def run_model(client: OpenAI, model: str, questions: list[dict], conn,
              pace_s: float) -> ModelReport:
    traces: list[CallTrace] = []
    for i, q in enumerate(questions):
        t = run_one_call(client, model, q, conn)
        traces.append(t)
        if i < len(questions) - 1 and pace_s:
            time.sleep(pace_s)

    n = len(traces)
    correct = sum(t.correct for t in traces)
    json_ok = sum(t.json_parsed_ok for t in traces)
    lats = [t.latency_ms for t in traces if t.latency_ms > 0]
    ttfts = [t.ttft_ms for t in traces if t.ttft_ms is not None]
    pt_mean = statistics.mean([t.prompt_tokens for t in traces]) if traces else 0.0
    ct_mean = statistics.mean([t.completion_tokens for t in traces]) if traces else 0.0

    rates = PRICING.get(model)
    cpq: float | None = None
    if rates and pt_mean and ct_mean:
        cpq = (pt_mean * rates["input"] + ct_mean * rates["output"]) / 1_000_000

    fatal = None
    notes: list[str] = []
    err_msgs = [t.error for t in traces if t.error]
    if err_msgs and len(err_msgs) == n:
        fatal = err_msgs[0]
        notes.append("every call failed identically — see error")
    if json_ok < n:
        notes.append(f"JSON-mode parsed on {json_ok}/{n} calls")
    if all(t.json_parsed_ok and not t.sql for t in traces):
        notes.append("returned valid JSON but with empty/null sql field")

    return ModelReport(
        model=model, n=n, correct=correct,
        accuracy=correct / n if n else 0.0,
        json_ok_count=json_ok, json_ok_rate=json_ok / n if n else 0,
        median_ttft_ms=statistics.median(ttfts) if ttfts else None,
        median_latency_ms=statistics.median(lats) if lats else 0.0,
        p90_latency_ms=_percentile(lats, 90),
        mean_prompt_tokens=pt_mean,
        mean_completion_tokens=ct_mean,
        cost_per_query_usd=cpq,
        pricing=rates,
        fatal_error=fatal,
        notes="; ".join(notes) or "—",
        traces=traces,
    )


def probe_model(client: OpenAI, model: str) -> dict[str, Any]:
    """One tiny call to confirm whether a model is reachable on this key."""
    extra = {"reasoning_effort": "none"} if model in REASONING_OFF else None
    try:
        client.with_options(timeout=40.0).chat.completions.create(
            model=model, messages=[{"role": "user", "content": "ok"}],
            temperature=0, max_tokens=8, extra_body=extra,
        )
        return {"model": model, "reachable": True, "error": None}
    except Exception as e:
        return {"model": model, "reachable": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def load_gold_questions(dev_path: str, synth_path: str) -> list[dict]:
    """Concat 10 dev + 15 synthetic into a single 25-question gold set."""
    dev = json.loads(Path(dev_path).read_text())
    synth = json.loads(Path(synth_path).read_text())
    return dev + synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(PRICING.keys()))
    ap.add_argument("--out", default="data/fireworks_sweep_v2.json")
    ap.add_argument("--dev",   default="data/dev_questions_with_answers.json")
    ap.add_argument("--synth", default="data/synthetic_questions.json")
    ap.add_argument("--db",    default="data/Chinook.db")
    ap.add_argument("--pace",  type=float, default=INTER_QUERY_SLEEP_S)
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on questions (debug)")
    ap.add_argument("--skip-probe", action="store_true",
                    help="skip the reachability probe of PROBE_ONLY models")
    args = ap.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")
    client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=api_key)

    questions = load_gold_questions(args.dev, args.synth)
    if args.limit:
        questions = questions[: args.limit]
    conn = load_db(args.db)

    print(f"Fireworks sweep · {len(args.models)} models × {len(questions)} questions")
    print(f"Pacing {args.pace}s between calls, {INTER_MODEL_SLEEP_S}s between models\n")

    reports: list[ModelReport] = []

    def checkpoint(probed):
        Path(args.out).write_text(json.dumps(
            {"platform": "fireworks", "base_url": FIREWORKS_BASE_URL,
             "methodology": "25-question gold set · streaming · TTFT captured · "
                            "one-shot, no repair, no retry · JSON mode · temp 0 · "
                            "max_tokens 1024 — same scored protocol as "
                            "data/baseten_sweep.json. Inter-call pacing raised to "
                            f"{args.pace}s (vs Baseten's 1.5s) to clear Fireworks "
                            "shared-serverless 429s so accuracy reflects the model.",
             "n_questions": len(questions),
             "inter_query_pace_s": args.pace,
             "probed_unavailable": probed,
             "reports": [asdict(rep) for rep in reports]},
            indent=2, ensure_ascii=False,
        ))

    for i, m in enumerate(args.models):
        short = m.split("/")[-1]
        print(f"[{i+1}/{len(args.models)}] {short} …", flush=True)
        t_model = time.perf_counter()
        try:
            r = run_model(client, m, questions, conn, args.pace)
        except Exception as e:
            print(f"  fatal: {e}")
            continue
        dt = time.perf_counter() - t_model
        cpq = f"${r.cost_per_query_usd:.6f}" if r.cost_per_query_usd is not None else "—"
        ttft = f"{r.median_ttft_ms:.0f}ms" if r.median_ttft_ms is not None else "—"
        print(
            f"  acc {r.correct}/{r.n} ({r.accuracy:.0%}) · "
            f"json {r.json_ok_count}/{r.n} · "
            f"ttft {ttft} · p50 {r.median_latency_ms:.0f}ms · "
            f"p90 {r.p90_latency_ms:.0f}ms · cost/q {cpq} · "
            f"took {dt:.0f}s"
        )
        print(f"  notes: {r.notes}")
        reports.append(r)
        # Checkpoint after each model so a mid-sweep failure doesn't lose work.
        checkpoint([])
        if i < len(args.models) - 1:
            time.sleep(INTER_MODEL_SLEEP_S)

    probed: list[dict] = []
    if not args.skip_probe and PROBE_ONLY:
        print("\nProbing models not in the sweep …")
        for m in PROBE_ONLY:
            time.sleep(INTER_QUERY_SLEEP_S)
            p = probe_model(client, m)
            status = "reachable" if p["reachable"] else p["error"]
            print(f"  {m.split('/')[-1]:18s} {status}")
            probed.append(p)

    checkpoint(probed)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
