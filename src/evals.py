"""Evaluation framework for text-to-SQL quality.

Comparison philosophy: a row is correct if its *values* (in column order) match
the gold row's values, regardless of how the model named its columns. We compare
result sets as multisets — order-insensitive — which is the standard convention
in text-to-SQL benchmarks (Spider, BIRD). Floats compare with a small epsilon
because aggregations frequently differ in trailing decimals.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from src.agent import SqlAgent, DEFAULT_MODEL, FIREWORKS_BASE_URL
from src.utils import load_db, query_db

load_dotenv()

FLOAT_ABS_EPS = 0.01      # currency rounded to cents
FLOAT_REL_EPS = 1e-3


def _normalize_value(v: Any) -> Any:
    """Canonicalize a single cell so int/float and trimmed-string compare cleanly."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, bool):  # bool is a subclass of int — keep it separate
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _values_close(a: Any, b: Any) -> bool:
    a, b = _normalize_value(a), _normalize_value(b)
    if isinstance(a, float) and isinstance(b, float):
        if a == b:
            return True
        return abs(a - b) <= max(FLOAT_ABS_EPS, FLOAT_REL_EPS * max(abs(a), abs(b)))
    return a == b


def _row_signature(row: dict[str, Any]) -> tuple:
    """Tuple of values in dict-insertion order; matches column order from sqlite."""
    return tuple(_normalize_value(v) for v in row.values())


def rows_match(predicted: list[dict] | None, expected: list[dict]) -> bool:
    """Multiset comparison of result rows; tolerant to column renames and float jitter."""
    if predicted is None:
        return False
    if len(predicted) != len(expected):
        return False

    # Same shape required (number of columns per row).
    if predicted and len(predicted[0]) != len(expected[0]):
        return False

    # Greedy multiset match on row signatures.
    remaining = [_row_signature(r) for r in expected]
    for p in predicted:
        sig = _row_signature(p)
        for i, e in enumerate(remaining):
            if len(sig) == len(e) and all(_values_close(x, y) for x, y in zip(sig, e)):
                remaining.pop(i)
                break
        else:
            return False
    return not remaining


@dataclass
class QResult:
    qid: str
    tier: int
    question: str
    correct: bool
    latency_ms: float
    repairs: int
    error: str | None
    sql: str | None
    rows: list[dict] | None


@dataclass
class EvalRun:
    model: str
    results: list[QResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.results) / len(self.results) if self.results else 0.0

    def by_tier(self) -> dict[int, tuple[int, int]]:
        out: dict[int, list[bool]] = {}
        for r in self.results:
            out.setdefault(r.tier, []).append(r.correct)
        return {t: (sum(v), len(v)) for t, v in out.items()}

    @property
    def median_latency_ms(self) -> float:
        ls = sorted(r.latency_ms for r in self.results)
        if not ls:
            return 0.0
        n = len(ls)
        return ls[n // 2] if n % 2 else (ls[n // 2 - 1] + ls[n // 2]) / 2


def evaluate(
    questions_path: str = "data/dev_questions_with_answers.json",
    db_path: str = "data/Chinook.db",
    model: str | None = None,
    label: str | None = None,
) -> EvalRun:
    questions = json.loads(Path(questions_path).read_text())
    conn = load_db(db_path)
    agent = SqlAgent(conn) if model is None else SqlAgent(conn, model=model)
    run = EvalRun(model=label or agent.model)

    for q in questions:
        t0 = time.perf_counter()
        out = agent.ask(q["question"])
        dt = (time.perf_counter() - t0) * 1000
        correct = rows_match(out.rows, q["expected_result"]) if out.error is None else False
        run.results.append(QResult(
            qid=q["id"], tier=q["tier"], question=q["question"],
            correct=correct, latency_ms=dt, repairs=out.repairs,
            error=out.error, sql=out.sql, rows=out.rows,
        ))
    return run


def print_run(run: EvalRun) -> None:
    print(f"\n=== {run.model} ===")
    print(f"{'qid':<8}{'tier':<6}{'ok':<4}{'lat ms':<9}{'rep':<5}{'note'}")
    print("-" * 80)
    for r in run.results:
        ok = "Y" if r.correct else "N"
        note = ""
        if r.error:
            note = f"ERROR: {r.error[:60]}"
        elif not r.correct:
            note = f"got {len(r.rows or [])} rows; first: {(r.rows or [{}])[0]}"[:60]
        print(f"{r.qid:<8}{r.tier:<6}{ok:<4}{r.latency_ms:<9.0f}{r.repairs:<5}{note}")
    print(f"\naccuracy: {run.accuracy:.0%}  ({sum(r.correct for r in run.results)}/{len(run.results)})")
    print(f"median latency: {run.median_latency_ms:.0f} ms")
    by_tier = run.by_tier()
    for t in sorted(by_tier):
        c, n = by_tier[t]
        print(f"  tier {t}: {c}/{n}")


def print_side_by_side(runs: list[EvalRun]) -> None:
    qids = [r.qid for r in runs[0].results]
    questions = {r.qid: r for r in runs[0].results}
    print("\n=== side-by-side ===")
    header = f"{'qid':<8}{'tier':<6}" + "".join(f"{r.model.split('/')[-1]:<28}" for r in runs)
    print(header)
    print("-" * len(header))
    for qid in qids:
        tier = questions[qid].tier
        cells = []
        for run in runs:
            r = next(x for x in run.results if x.qid == qid)
            mark = "Y" if r.correct else ("E" if r.error else "N")
            cells.append(f"{mark} {r.latency_ms:>5.0f}ms  rep={r.repairs}".ljust(28))
        print(f"{qid:<8}{tier:<6}" + "".join(cells))
    print("-" * len(header))
    summary = f"{'accuracy':<14}" + "".join(f"{run.accuracy:.0%} ({sum(x.correct for x in run.results)}/{len(run.results)})".ljust(28) for run in runs)
    print(summary)
    print(f"{'median lat':<14}" + "".join(f"{run.median_latency_ms:>5.0f} ms".ljust(28) for run in runs))
    for tier in sorted({r.tier for r in runs[0].results}):
        line = f"  tier {tier:<8}"
        for run in runs:
            c = sum(1 for x in run.results if x.tier == tier and x.correct)
            n = sum(1 for x in run.results if x.tier == tier)
            line += f"{c}/{n}".ljust(28)
        print(line)


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def format_answer_summary(rows: list[dict] | None, max_rows: int = 10) -> str:
    """Render a result set as a single-line human-readable string.

    Heuristic: take up to two leading string columns as the entity name
    (handles FirstName+LastName), put the rest in parens as "key=value".
    """
    if not rows:
        return "(no rows)"
    truncated = len(rows) > max_rows
    chunk = rows[:max_rows]
    cols = list(chunk[0].keys())

    if len(cols) == 1:
        parts = [_fmt_value(r[cols[0]]) for r in chunk]
        out = ", ".join(parts)
    else:
        primary_cols: list[str] = []
        for c in cols:
            if len(primary_cols) >= 2:
                break
            if isinstance(chunk[0][c], str):
                primary_cols.append(c)
            else:
                break
        if not primary_cols:
            primary_cols = [cols[0]]
        rest_cols = [c for c in cols if c not in primary_cols]
        parts = []
        for r in chunk:
            primary = " ".join(_fmt_value(r[c]) for c in primary_cols)
            if rest_cols:
                rest = ", ".join(f"{c}={_fmt_value(r[c])}" for c in rest_cols)
                parts.append(f"{primary} ({rest})")
            else:
                parts.append(primary)
        out = "; ".join(parts)

    if truncated:
        out += f"; ... ({len(rows) - max_rows} more)"
    return out


def write_dev_answers(run: EvalRun, path: str) -> None:
    out: dict[str, dict[str, str]] = {}
    for r in run.results:
        out[r.qid] = {
            "sql": r.sql or "",
            "answer": format_answer_summary(r.rows),
        }
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Baseline: the customer's current approach.
#
# Raul's email shows their prototype prompt is literally
# `Convert this question to SQL: {question}` — no schema, no JSON mode,
# no retry, no execute-and-repair. We replay that against Kimi K2.6 to get
# an apples-to-apples "what would the same model score with their prompt"
# number — separating the lift from prompt engineering vs. raw model
# capability.
# ---------------------------------------------------------------------------
BASELINE_PROMPT_TEMPLATE = "Convert this question to SQL:\n{question}"

_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_PREFIX_RE = re.compile(r"^\s*(SQL|Query|Answer)\s*:\s*", re.IGNORECASE)


def _extract_sql_naive(text: str) -> str:
    """Pull a SQL string out of a free-form model response.

    Handles fenced ```sql blocks (most common) and bare SQL with a
    "SQL:" prefix. Strips trailing semicolons.
    """
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(";").strip()
    return _PREFIX_RE.sub("", text).rstrip(";").strip()


def _baseline_call(client: OpenAI, model: str, question: str) -> str:
    """One naive Fireworks call. Includes 429 retry so eval is fair."""
    delays = [2.0, 4.0, 8.0]
    for attempt in range(len(delays) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": BASELINE_PROMPT_TEMPLATE.format(question=question),
                }],
                temperature=0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt >= len(delays):
                raise
            time.sleep(delays[attempt] + random.uniform(0, 0.5))
    return ""  # unreachable


def evaluate_baseline(
    questions_path: str = "data/dev_questions_with_answers.json",
    db_path: str = "data/Chinook.db",
    model: str = "accounts/fireworks/models/kimi-k2p6",
) -> EvalRun:
    questions = json.loads(Path(questions_path).read_text())
    conn = load_db(db_path)
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set.")
    client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=api_key)

    run = EvalRun(model=f"{model} (baseline-prompt)")
    for q in questions:
        t0 = time.perf_counter()
        sql: str | None = None
        rows: list[dict] | None = None
        error: str | None = None
        try:
            raw = _baseline_call(client, model, q["question"])
            sql = _extract_sql_naive(raw) or None
            if sql:
                rows = query_db(conn, sql, return_as_df=False)
            else:
                error = "model returned no extractable SQL"
        except Exception as e:
            error = str(e)
        dt = (time.perf_counter() - t0) * 1000

        correct = (
            rows_match(rows, q["expected_result"])
            if error is None and rows is not None else False
        )
        run.results.append(QResult(
            qid=q["id"], tier=q["tier"], question=q["question"],
            correct=correct, latency_ms=dt, repairs=0,
            error=error, sql=sql, rows=rows,
        ))
    return run


def write_baseline_results(run: EvalRun, path: str) -> None:
    """Enriched JSON output: SQL, summary, correctness, error per question."""
    out: dict[str, Any] = {}
    for r in run.results:
        entry = {
            "tier": r.tier,
            "question": r.question,
            "sql": r.sql or "",
            "answer": format_answer_summary(r.rows),
            "correct": r.correct,
        }
        if r.error:
            entry["error"] = r.error
        out[r.qid] = entry
    out["_summary"] = {
        "model": run.model,
        "approach": "baseline-prompt — 'Convert this question to SQL: {question}'",
        "accuracy": f"{sum(x.correct for x in run.results)}/{len(run.results)}",
        "median_latency_ms": run.median_latency_ms,
        "by_tier": {str(t): f"{c}/{n}" for t, (c, n) in run.by_tier().items()},
    }
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="python -m src.evals")
    ap.add_argument("models", nargs="*", help="Fireworks model ids (default: agent default)")
    ap.add_argument("--questions", default="data/dev_questions_with_answers.json",
                    help="path to a question file in the dev-set schema")
    ap.add_argument("--out-answers", help="write answers JSON from the first model's run")
    ap.add_argument("--baseline", action="store_true",
                    help="run the customer's naive 'Convert this question to SQL' prompt "
                         "(no schema, no JSON mode, no repair) instead of the agent")
    ap.add_argument("--out-baseline", help="write enriched baseline results JSON to this path")
    args = ap.parse_args()

    if args.baseline:
        model = args.models[0] if args.models else "accounts/fireworks/models/kimi-k2p6"
        run = evaluate_baseline(questions_path=args.questions, model=model)
        print_run(run)
        out_path = args.out_baseline or "data/baseline_results.json"
        write_baseline_results(run, out_path)
        print(f"\nwrote {out_path}")
    else:
        runs: list[EvalRun] = []
        targets = args.models or [None]
        for m in targets:
            run = evaluate(model=m, label=m, questions_path=args.questions)
            print_run(run)
            runs.append(run)
        if len(runs) > 1:
            print_side_by_side(runs)
        if args.out_answers and runs:
            write_dev_answers(runs[0], args.out_answers)
            print(f"\nwrote {args.out_answers}")
