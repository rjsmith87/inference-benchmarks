"""Single-question stress test across every model in the dashboard dropdown.

Question: "How many albums are in the database?"  Gold answer: 347.
Per-call timeout: 15 seconds. No retries, no repair loop — first-shot.

Captures, per model:
  - responded (true if HTTP returned without timeout/error)
  - latency_ms
  - json_mode_honored (response parsed cleanly as JSON object with `sql`)
  - sql_extracted (any SQL the executor could try, fenced or otherwise)
  - sql_executed (SQLite ran without raising)
  - answer_correct (any cell == 347 in the result set)
  - error / fallback notes

Usage:
    PYTHONPATH=. python scripts/single_q_test.py --out data/single_q_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

for ln in Path(".env").read_text().splitlines():
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k, v.strip())

from openai import OpenAI

from src.agent import (
    SYSTEM_PROMPT, FIREWORKS_BASE_URL,
    _format_schema_compact, _trim_schema_for_question,
)
from src.utils import load_db, query_db


QUESTION = "How many albums are in the database?"
GOLD = 347
TIMEOUT_S = 15.0
PACE_S = 4.0  # tighter pacing to stay under the per-minute QPS cap

MODELS = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/kimi-k2p5",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/minimax-m2p7",
    "accounts/fireworks/models/qwen3-8b",
]


@dataclass
class ModelResult:
    model: str
    responded: bool
    latency_ms: float
    raw_content_preview: str
    json_mode_honored: bool
    sql_extracted: bool
    sql_text: str | None
    sql_executed: bool
    answer_correct: bool
    answer_value: Any
    error: str | None
    notes: str


_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql_loose(text: str) -> str | None:
    """Pull SQL out of free-form output if JSON mode was ignored."""
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        cand = m.group(1).strip().rstrip(";").strip()
        return cand or None
    # Last resort: leading 'SELECT'
    m2 = re.search(r"(SELECT\s+[\s\S]+?)(?:;|\n\n|$)", text, re.IGNORECASE)
    return m2.group(1).strip().rstrip(";").strip() if m2 else None


def test_one(client: OpenAI, model: str, conn) -> ModelResult:
    schema = _trim_schema_for_question(conn, QUESTION, _format_schema_compact(conn), compact=True)
    sys_msg = SYSTEM_PROMPT.format(schema=schema)
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": QUESTION},
    ]
    extra: dict[str, Any] = {}
    if "qwen3" in model:
        extra["reasoning_effort"] = "none"

    t0 = time.perf_counter()
    try:
        resp = client.with_options(timeout=TIMEOUT_S).chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_object"},
            temperature=0, max_tokens=512,
            extra_body=extra or None,
        )
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        return ModelResult(
            model=model, responded=False, latency_ms=dt,
            raw_content_preview="", json_mode_honored=False,
            sql_extracted=False, sql_text=None, sql_executed=False,
            answer_correct=False, answer_value=None,
            error=msg,
            notes="timeout" if "Timeout" in msg or "timeout" in msg else "API error",
        )

    dt = (time.perf_counter() - t0) * 1000
    content = resp.choices[0].message.content or ""
    preview = content[:240]

    json_ok = False
    sql_text: str | None = None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "sql" in parsed:
            json_ok = True
            sql_text = (parsed.get("sql") or "").strip().rstrip(";").strip() or None
    except json.JSONDecodeError:
        pass

    if not sql_text:
        sql_text = extract_sql_loose(content)

    sql_extracted = bool(sql_text)
    executed = False
    correct = False
    answer_val: Any = None
    err: str | None = None

    if sql_extracted:
        try:
            rows = query_db(conn, sql_text, return_as_df=False)
            executed = True
            for r in rows:
                for v in r.values():
                    try:
                        if int(v) == GOLD:
                            correct = True
                            answer_val = int(v)
                            break
                    except (TypeError, ValueError):
                        pass
                if correct:
                    break
            if not correct and rows:
                first = next(iter(rows[0].values()), None)
                answer_val = first
        except Exception as e:
            err = f"sqlite: {str(e)[:150]}"

    notes_parts: list[str] = []
    if not json_ok and sql_extracted:
        notes_parts.append("JSON-mode ignored — fell back to free-text/fence extraction")
    if not sql_extracted:
        notes_parts.append("no SQL extractable from response")
    if executed and not correct:
        notes_parts.append(f"executed but answer={answer_val} (gold {GOLD})")
    if correct:
        notes_parts.append("correct")
    return ModelResult(
        model=model, responded=True, latency_ms=dt,
        raw_content_preview=preview,
        json_mode_honored=json_ok,
        sql_extracted=sql_extracted, sql_text=sql_text,
        sql_executed=executed, answer_correct=correct, answer_value=answer_val,
        error=err, notes="; ".join(notes_parts) or "—",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/single_q_test.json")
    args = ap.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        sys.exit("FIREWORKS_API_KEY not set")
    client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=api_key)
    conn = load_db("data/Chinook.db")

    print(f"Q: {QUESTION!r}")
    print(f"Gold answer: {GOLD}")
    print(f"Per-call timeout: {TIMEOUT_S}s, pacing: {PACE_S}s\n")

    results: list[ModelResult] = []
    for i, m in enumerate(MODELS, 1):
        short = m.split("/")[-1]
        print(f"[{i}/{len(MODELS)}] {short} … ", end="", flush=True)
        r = test_one(client, m, conn)
        results.append(r)
        flag = (
            "✓ correct" if r.answer_correct else
            "✗ executed but wrong" if r.sql_executed else
            "✗ no SQL" if r.responded else
            "✗ no response"
        )
        print(f"{flag}  ({r.latency_ms:.0f}ms · json={'Y' if r.json_mode_honored else 'N'})")
        if r.error:
            print(f"     err: {r.error[:120]}")
        if r.sql_text and not r.answer_correct:
            print(f"     sql: {r.sql_text[:100]}")
        if i < len(MODELS):
            time.sleep(PACE_S)

    Path(args.out).write_text(json.dumps(
        {"question": QUESTION, "gold": GOLD, "timeout_s": TIMEOUT_S,
         "results": [asdict(r) for r in results]},
        indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
