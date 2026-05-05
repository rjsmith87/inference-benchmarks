"""Latency + cost benchmarking for the text-to-SQL agent.

Runs the agent across a question set, captures token counts from the Fireworks
response, and produces per-query and aggregate stats.

Pricing values are taken from fireworks.ai/pricing and the customer-cited
GPT-5.4 rate. Update when the published rates change.

Run:
  python -m src.perf                                   # default model
  python -m src.perf <model_id> [<model_id> ...]       # one or more models
  python -m src.perf <model_id> --out perf.json        # write JSON summary
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from src.agent import SqlAgent, DEFAULT_MODEL
from src.evals import rows_match
from src.utils import load_db

# USD per 1M tokens. Refresh from fireworks.ai/pricing if rates change.
PRICING_USD_PER_M: dict[str, dict[str, float]] = {
    "accounts/fireworks/models/kimi-k2p6": {"input": 0.95, "output": 4.00},
    "accounts/fireworks/models/qwen3-8b":  {"input": 0.20, "output": 0.20},   # ESTIMATE
    # Customer baseline rate from Raul's email (GPT-5.4 published pricing).
    "_baseline_proprietary":               {"input": 2.50, "output": 15.00},
}

CUSTOMER_VOLUME_PER_DAY = 30_000  # 1000 users × 30 queries/day from the Raul email
PACE_BETWEEN_QUERIES_S = 2.0       # avoid tripping per-minute rate limits during the run


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_USD_PER_M.get(model)
    if rates is None:
        return 0.0
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


@dataclass
class PerfRow:
    qid: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    correct: bool
    repairs: int
    error: str | None


@dataclass
class PerfReport:
    model: str
    n: int
    accuracy: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_cost_usd: float
    daily_cost_at_volume_usd: float
    monthly_cost_at_volume_usd: float
    rows: list[PerfRow] = field(default_factory=list)


def measure(
    model: str | None = None,
    questions_path: str = "data/dev_questions_with_answers.json",
    db_path: str = "data/Chinook.db",
    pace_s: float = PACE_BETWEEN_QUERIES_S,
) -> PerfReport:
    qs = json.loads(Path(questions_path).read_text())
    conn = load_db(db_path)
    agent = SqlAgent(conn) if model is None else SqlAgent(conn, model=model)

    rows: list[PerfRow] = []
    for i, q in enumerate(qs):
        if i > 0 and pace_s:
            time.sleep(pace_s)
        t0 = time.perf_counter()
        out = agent.ask(q["question"])
        dt = (time.perf_counter() - t0) * 1000
        cost = cost_for(agent.model, out.prompt_tokens, out.completion_tokens)
        correct = rows_match(out.rows, q["expected_result"]) if out.error is None else False
        rows.append(PerfRow(
            qid=q["id"], latency_ms=dt, prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens, cost_usd=cost,
            correct=correct, repairs=out.repairs, error=out.error,
        ))

    lats = [r.latency_ms for r in rows]
    costs = [r.cost_usd for r in rows]
    mean_cost = statistics.mean(costs) if costs else 0.0
    return PerfReport(
        model=agent.model,
        n=len(rows),
        accuracy=sum(r.correct for r in rows) / len(rows) if rows else 0.0,
        p50_ms=percentile(lats, 50),
        p90_ms=percentile(lats, 90),
        p99_ms=percentile(lats, 99),
        mean_prompt_tokens=statistics.mean([r.prompt_tokens for r in rows]) if rows else 0,
        mean_completion_tokens=statistics.mean([r.completion_tokens for r in rows]) if rows else 0,
        mean_cost_usd=mean_cost,
        daily_cost_at_volume_usd=mean_cost * CUSTOMER_VOLUME_PER_DAY,
        monthly_cost_at_volume_usd=mean_cost * CUSTOMER_VOLUME_PER_DAY * 30,
        rows=rows,
    )


def print_report(rep: PerfReport) -> None:
    short = rep.model.split("/")[-1]
    print(f"\n=== {short} ===")
    print(f"  n={rep.n}  accuracy={rep.accuracy:.0%}")
    print(f"  latency  p50={rep.p50_ms:.0f}ms  p90={rep.p90_ms:.0f}ms  p99={rep.p99_ms:.0f}ms")
    print(f"  tokens   prompt~={rep.mean_prompt_tokens:.0f}  completion~={rep.mean_completion_tokens:.0f}")
    print(f"  cost     ${rep.mean_cost_usd:.6f}/query  (${rep.mean_cost_usd*1000:.3f} per 1k queries)")
    print(f"  @ {CUSTOMER_VOLUME_PER_DAY:,} q/day → ${rep.daily_cost_at_volume_usd:,.2f}/day  ${rep.monthly_cost_at_volume_usd:,.0f}/mo")


def print_baseline_compare(reps: list[PerfReport]) -> None:
    base = PRICING_USD_PER_M.get("_baseline_proprietary")
    if not base or not reps:
        return
    proxy = reps[0]
    base_cost = (proxy.mean_prompt_tokens * base["input"]
                 + proxy.mean_completion_tokens * base["output"]) / 1_000_000
    base_daily = base_cost * CUSTOMER_VOLUME_PER_DAY
    print("\n--- vs proprietary baseline (GPT-5.4 published rates) ---")
    print(f"baseline cost/query: ${base_cost:.6f}  daily: ${base_daily:,.2f}  monthly: ${base_daily*30:,.0f}")
    print(f"{'model':<28}{'$/query':>12}{'Δ vs baseline':>20}")
    for r in reps:
        delta = (1 - r.mean_cost_usd / base_cost) * 100 if base_cost else 0
        short = r.model.split("/")[-1]
        print(f"{short:<28}{r.mean_cost_usd:>12.6f}{delta:>18.1f}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.perf")
    ap.add_argument("models", nargs="*", help="Fireworks model ids (default: agent default)")
    ap.add_argument("--out", help="write JSON summary to this path")
    ap.add_argument("--pace", type=float, default=PACE_BETWEEN_QUERIES_S,
                    help="seconds to sleep between queries (default: avoids 429s)")
    args = ap.parse_args(argv)

    targets = args.models or [DEFAULT_MODEL]
    reports: list[PerfReport] = []
    for m in targets:
        rep = measure(model=m, pace_s=args.pace)
        print_report(rep)
        reports.append(rep)

    if reports:
        print_baseline_compare(reports)

    if args.out:
        payload = {"reports": [asdict(r) for r in reports]}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
