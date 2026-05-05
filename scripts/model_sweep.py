"""Run the 10-question dev set across multiple Fireworks models.

For each model, captures accuracy, latency, cost (when pricing known), and a
JSON-reliability signal: did the response come back as a JSON object the
parser could read on the first call? Models that silently ignore
`response_format={"type": "json_object"}` show up as parse failures.

Usage:
    PYTHONPATH=. python scripts/model_sweep.py
    PYTHONPATH=. python scripts/model_sweep.py --out data/model_sweep.json
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

from openai import OpenAI, APIError, RateLimitError

from src.agent import (
    SqlAgent, SYSTEM_PROMPT, FIREWORKS_BASE_URL,
    _format_schema_compact, _trim_schema_for_question,
)
from src.evals import rows_match
from src.utils import load_db


# Pricing per 1M tokens (USD). Verified rates only; otherwise None.
# Kimi K2.6 from fireworks.ai/pricing; Qwen3-8B from prior estimate (kept for reference).
PRICING: dict[str, dict[str, float] | None] = {
    "accounts/fireworks/models/kimi-k2p6":     {"input": 0.95, "output": 4.00},
    # Below: rates need verification before publishing customer-facing numbers.
    "accounts/fireworks/models/kimi-k2p5":     None,
    "accounts/fireworks/models/deepseek-v4-pro": None,
    "accounts/fireworks/models/glm-5":         None,
    "accounts/fireworks/models/glm-5p1":       None,
    "accounts/fireworks/models/minimax-m2p7":  None,
    "accounts/fireworks/models/qwen3-8b":      {"input": 0.20, "output": 0.20},  # est
}


@dataclass
class CallTrace:
    qid: str
    raw_content: str
    json_parsed_ok: bool
    sql: str | None
    rows: list[dict] | None
    correct: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None


@dataclass
class ModelReport:
    model: str
    n: int
    correct: int
    json_ok_count: int
    json_ok_rate: float
    median_latency_ms: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    cost_per_query_usd: float | None
    fatal_error: str | None
    notes: str
    traces: list[CallTrace] = field(default_factory=list)


def run_one_call(client: OpenAI, model: str, question: str, conn) -> CallTrace:
    """One agent call with both JSON-mode signal and execute-and-compare.

    Mirrors agent.ask() but tracks the JSON-parse-success signal explicitly so
    models that ignore `response_format` are visible in the report.
    """
    schema = _trim_schema_for_question(conn, question, _format_schema_compact(conn), compact=True)
    sys_msg = SYSTEM_PROMPT.format(schema=schema)
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": question},
    ]
    extra: dict[str, Any] = {}
    if "qwen3" in model:
        extra["reasoning_effort"] = "none"

    t0 = time.perf_counter()
    try:
        resp = client.with_options(timeout=30.0).chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_object"},
            temperature=0, max_tokens=1024,
            extra_body=extra or None,
        )
    except (APIError, RateLimitError, Exception) as e:
        return CallTrace(
            qid="", raw_content="", json_parsed_ok=False,
            sql=None, rows=None, correct=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=0, completion_tokens=0,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
    latency = (time.perf_counter() - t0) * 1000
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0

    json_ok = False
    sql: str | None = None
    parsed: dict[str, Any] = {}
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
            from src.utils import query_db
            rows = query_db(conn, sql, return_as_df=False)
        except Exception as e:
            err = f"sqlite: {e}"
    return CallTrace(
        qid="", raw_content=content[:500], json_parsed_ok=json_ok,
        sql=sql, rows=rows, correct=False,
        latency_ms=latency, prompt_tokens=pt, completion_tokens=ct,
        error=err,
    )


def run_model(model: str, questions: list[dict], conn) -> ModelReport:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")
    client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=api_key)

    traces: list[CallTrace] = []
    for q in questions:
        t = run_one_call(client, model, q["question"], conn)
        t.qid = q["id"]
        if t.rows is not None and not t.error:
            t.correct = rows_match(t.rows, q["expected_result"])
        traces.append(t)
        time.sleep(1.5)  # pace; avoid 429s within the sweep

    n = len(traces)
    correct = sum(t.correct for t in traces)
    json_ok = sum(t.json_parsed_ok for t in traces)
    lats = sorted(t.latency_ms for t in traces if t.latency_ms > 0)
    p50 = lats[len(lats) // 2] if lats else 0.0
    pt_mean = statistics.mean([t.prompt_tokens for t in traces]) if traces else 0.0
    ct_mean = statistics.mean([t.completion_tokens for t in traces]) if traces else 0.0

    rates = PRICING.get(model)
    cpq: float | None = None
    if rates and pt_mean and ct_mean:
        cpq = (pt_mean * rates["input"] + ct_mean * rates["output"]) / 1_000_000

    fatal = None
    notes_parts: list[str] = []
    err_messages = [t.error for t in traces if t.error]
    if err_messages and len(err_messages) == n:
        fatal = err_messages[0]
        notes_parts.append("every call failed identically — see error")
    if json_ok < n:
        notes_parts.append(f"JSON-mode honored on {json_ok}/{n} calls")
    if all(t.json_parsed_ok and not t.sql for t in traces):
        notes_parts.append("returned valid JSON but with empty/null sql field")

    return ModelReport(
        model=model, n=n, correct=correct,
        json_ok_count=json_ok, json_ok_rate=json_ok / n if n else 0,
        median_latency_ms=p50,
        mean_prompt_tokens=pt_mean, mean_completion_tokens=ct_mean,
        cost_per_query_usd=cpq,
        fatal_error=fatal,
        notes="; ".join(notes_parts) or "—",
        traces=traces,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/kimi-k2p5",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/glm-5",
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/minimax-m2p7",
    ])
    ap.add_argument("--out", default="data/model_sweep.json")
    ap.add_argument("--questions", default="data/dev_questions_with_answers.json")
    ap.add_argument("--db", default="data/Chinook.db")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    conn = load_db(args.db)

    print(f"Sweeping {len(args.models)} models on {len(questions)} dev questions")
    print(f"Pacing 1.5s between calls; 8s between models.\n")

    reports: list[ModelReport] = []
    for i, m in enumerate(args.models):
        short = m.split("/")[-1]
        print(f"[{i+1}/{len(args.models)}] {short} …", flush=True)
        try:
            r = run_model(m, questions, conn)
        except Exception as e:
            print(f"  fatal: {e}")
            continue
        reports.append(r)
        cost = f"${r.cost_per_query_usd:.6f}" if r.cost_per_query_usd is not None else "—"
        print(f"  {r.correct}/{r.n} correct · JSON {r.json_ok_count}/{r.n} "
              f"· p50 {r.median_latency_ms:.0f}ms · cost/q {cost}")
        print(f"  notes: {r.notes}")
        if i < len(args.models) - 1:
            time.sleep(8)

    Path(args.out).write_text(json.dumps(
        {"reports": [asdict(r) for r in reports]},
        indent=2, ensure_ascii=False,
    ))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
