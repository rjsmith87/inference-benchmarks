"""Concurrency benchmark for Baseten Model API.

Fires the SqlAgent (with repair loop — matches what would actually ship in
production) at varying concurrency levels against the top 3 Baseten models
from the model sweep. Captures per-call latency to show how P50 / P90 / P99
move as the platform absorbs concurrent load.

Why this matters: the headline TTFT / median latency from a single-call sweep
hides queue depth and prefill contention. At concurrency=10 on a shared
serverless tier you find out whether the platform actually scales or quietly
serializes.

Usage:
    PYTHONPATH=. python scripts/baseten_perf.py
    PYTHONPATH=. python scripts/baseten_perf.py --concurrencies 1 5 10 20
    PYTHONPATH=. python scripts/baseten_perf.py --out data/baseten_perf.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Manually load .env (avoids dotenv heredoc-frame quirk).
for ln in Path(".env").read_text().splitlines():
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k, v.strip())

from src.agent import SqlAgent
from src.evals import rows_match
from src.utils import load_db

BASETEN_BASE_URL = "https://inference.baseten.co/v1"

# Top 3 from the baseten_sweep accuracy ranking. Hold constant across runs.
DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2.6",
]
DEFAULT_CONCURRENCIES = [1, 5, 10]

# Per-thread DB connection cache — sqlite3.Connection isn't safe across threads,
# so each worker opens its own. Cheap (a few ms) but worth not repeating per call.
_TL = threading.local()


def _agent_for_thread(model: str, db_path: str) -> SqlAgent:
    if not hasattr(_TL, "agents"):
        _TL.agents = {}
    if model in _TL.agents:
        return _TL.agents[model]
    conn = load_db(db_path)
    agent = SqlAgent(conn, model=model, base_url=BASETEN_BASE_URL)
    _TL.agents[model] = agent
    return agent


@dataclass
class CallRow:
    qid: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    repairs: int
    correct: bool
    error: str | None
    started_at: float          # epoch seconds, relative to run start
    finished_at: float


@dataclass
class RunStats:
    model: str
    concurrency: int
    n: int
    completed: int             # excludes hard errors with no latency
    errors: int
    accuracy: float
    wallclock_s: float         # total time from first dispatch to last finish
    throughput_qps: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    rows: list[CallRow] = field(default_factory=list)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def run_one_call(model: str, q: dict, db_path: str, t_origin: float) -> CallRow:
    agent = _agent_for_thread(model, db_path)
    t0 = time.perf_counter()
    started = time.time() - t_origin
    try:
        out = agent.ask(q["question"])
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return CallRow(
            qid=q["id"], latency_ms=dt, prompt_tokens=0, completion_tokens=0,
            repairs=0, correct=False, error=f"{type(e).__name__}: {str(e)[:150]}",
            started_at=started, finished_at=time.time() - t_origin,
        )
    dt = (time.perf_counter() - t0) * 1000
    correct = False
    if out.rows is not None and not out.error:
        try:
            correct = rows_match(out.rows, q["expected_result"])
        except Exception:
            correct = False
    return CallRow(
        qid=q["id"], latency_ms=dt,
        prompt_tokens=out.prompt_tokens, completion_tokens=out.completion_tokens,
        repairs=out.repairs, correct=correct, error=out.error,
        started_at=started, finished_at=time.time() - t_origin,
    )


def run_model_at_concurrency(model: str, questions: list[dict], concurrency: int,
                              db_path: str) -> RunStats:
    # Reset per-thread agent cache between (model, concurrency) sweeps — keeps
    # threads idle to GC and avoids leaking state when the executor scales down.
    _TL.agents = {}
    t_origin = time.time()
    t0 = time.perf_counter()
    rows: list[CallRow] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(run_one_call, model, q, db_path, t_origin): q for q in questions}
        for f in as_completed(futures):
            rows.append(f.result())
    wallclock = time.perf_counter() - t0

    completed = [r for r in rows if not r.error]
    errors = [r for r in rows if r.error]
    lats_ok = [r.latency_ms for r in completed]
    n = len(rows)
    return RunStats(
        model=model,
        concurrency=concurrency,
        n=n,
        completed=len(completed),
        errors=len(errors),
        accuracy=sum(r.correct for r in rows) / n if n else 0.0,
        wallclock_s=wallclock,
        throughput_qps=n / wallclock if wallclock > 0 else 0.0,
        p50_ms=_percentile(lats_ok, 50),
        p90_ms=_percentile(lats_ok, 90),
        p99_ms=_percentile(lats_ok, 99),
        mean_ms=statistics.mean(lats_ok) if lats_ok else 0.0,
        rows=sorted(rows, key=lambda r: r.started_at),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--concurrencies", nargs="*", type=int, default=DEFAULT_CONCURRENCIES)
    ap.add_argument("--out", default="data/baseten_perf.json")
    ap.add_argument("--dev",   default="data/dev_questions_with_answers.json")
    ap.add_argument("--synth", default="data/synthetic_questions.json")
    ap.add_argument("--db",    default="data/Chinook.db")
    ap.add_argument("--cooldown", type=float, default=5.0,
                    help="seconds to sleep between (model, concurrency) runs")
    args = ap.parse_args()

    dev = json.loads(Path(args.dev).read_text())
    synth = json.loads(Path(args.synth).read_text())
    questions = dev + synth

    print(f"Baseten perf · {len(args.models)} models × {len(args.concurrencies)} concurrencies × {len(questions)} questions")
    print(f"  models: {', '.join(m.split('/')[-1] for m in args.models)}")
    print(f"  concurrencies: {args.concurrencies}")
    print(f"  cooldown {args.cooldown}s between (model, concurrency) runs\n")

    all_runs: list[RunStats] = []
    for model in args.models:
        short = model.split("/")[-1]
        for c in args.concurrencies:
            print(f"[{short} · c={c}] running …", flush=True)
            try:
                stats = run_model_at_concurrency(model, questions, c, args.db)
            except Exception as e:
                print(f"  fatal: {e}")
                continue
            print(f"  done · n={stats.n} · err={stats.errors} · acc={stats.accuracy:.0%} "
                  f"· wall={stats.wallclock_s:.1f}s · qps={stats.throughput_qps:.2f} "
                  f"· p50={stats.p50_ms:.0f}ms · p90={stats.p90_ms:.0f}ms · p99={stats.p99_ms:.0f}ms")
            all_runs.append(stats)
            # Checkpoint after every run.
            Path(args.out).write_text(json.dumps(
                {"platform": "baseten", "base_url": BASETEN_BASE_URL,
                 "runs": [asdict(r) for r in all_runs]},
                indent=2, ensure_ascii=False,
            ))
            time.sleep(args.cooldown)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
