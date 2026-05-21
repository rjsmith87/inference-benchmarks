"""Run the full agent (compact schema · JSON mode · execute-and-repair) over the
25-question gold set and write a structured result file for the dashboard.

This is the agent's *end-to-end* quality measurement — distinct from the
one-shot model sweeps (scripts/fireworks_sweep.py / baseten_sweep.py), which
deliberately disable the repair loop to isolate raw model capability. The
Quality tab sources its score-board, eval log, and failure analysis from the
JSON this writes.

Usage:
    PYTHONPATH=. python scripts/agent_eval.py
    PYTHONPATH=. python scripts/agent_eval.py --model accounts/fireworks/models/kimi-k2p6
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from src.evals import evaluate


def main() -> int:
    ap = argparse.ArgumentParser(prog="python scripts/agent_eval.py")
    ap.add_argument("--model", default="accounts/fireworks/models/kimi-k2p6")
    ap.add_argument("--dev",   default="data/dev_questions_with_answers.json")
    ap.add_argument("--synth", default="data/synthetic_questions.json")
    ap.add_argument("--db",    default="data/Chinook.db")
    ap.add_argument("--out",   default="data/agent_eval.json")
    args = ap.parse_args()

    print(f"Agent eval · {args.model}")
    print("Full agent: compact schema, JSON mode, execute-and-repair (max 2 turns)\n")

    results: list[dict] = []
    for setname, path in (("dev", args.dev), ("synth", args.synth)):
        print(f"[{setname}] {path} …", flush=True)
        run = evaluate(questions_path=path, db_path=args.db,
                       model=args.model, label=args.model)
        for r in run.results:
            ok = "Y" if r.correct else ("E" if r.error else "N")
            print(f"  {r.qid:<8} tier {r.tier}  {ok}  {r.latency_ms:>6.0f}ms  rep={r.repairs}")
            results.append({
                "qid": r.qid, "set": setname, "tier": r.tier,
                "question": r.question, "correct": bool(r.correct),
                "repairs": r.repairs, "latency_ms": round(r.latency_ms),
                "sql": r.sql, "error": r.error,
            })

    def tally(setname: str) -> dict:
        rs = [r for r in results if r["set"] == setname]
        return {"correct": sum(r["correct"] for r in rs), "total": len(rs)}

    dev_t, syn_t = tally("dev"), tally("synth")
    total_correct = dev_t["correct"] + syn_t["correct"]
    total_n = dev_t["total"] + syn_t["total"]

    by_tier: dict[str, dict] = {}
    for r in results:
        b = by_tier.setdefault(str(r["tier"]), {"correct": 0, "total": 0})
        b["correct"] += r["correct"]
        b["total"] += 1

    lat = sorted(r["latency_ms"] for r in results)

    out = {
        "model": args.model,
        "harness": "SqlAgent — compact schema-in-prompt, JSON mode, "
                   "execute-and-repair (max 2 repair turns), temperature 0",
        "n_questions": total_n,
        "dev": dev_t,
        "synth": syn_t,
        "combined": {
            "correct": total_correct, "total": total_n,
            "accuracy": total_correct / total_n if total_n else 0.0,
        },
        "by_tier": by_tier,
        "median_latency_ms": statistics.median(lat) if lat else 0.0,
        "total_repairs": sum(r["repairs"] for r in results),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\ndev {dev_t['correct']}/{dev_t['total']} · "
          f"synth {syn_t['correct']}/{syn_t['total']} · "
          f"combined {total_correct}/{total_n} "
          f"({out['combined']['accuracy']:.0%}) · "
          f"{out['total_repairs']} repair turns total")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
