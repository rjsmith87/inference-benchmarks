"""Interactive text-to-SQL CLI. Run with: python -m src.cli

Env vars:
  FIREWORKS_API_KEY  required
  FIREWORKS_MODEL    optional model override (default: agent.DEFAULT_MODEL)
  CHINOOK_DB         optional db path (default: data/Chinook.db)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.agent import SqlAgent
from src.utils import load_db, print_table_schema

load_dotenv()

HELP = """
Commands:
  exit, quit, q   end the session
  schema          show the database schema
  reset           start a fresh conversation (forget prior turns)
  help            show this message

Otherwise, type a natural-language question and press enter.
""".strip()

# Cap how much prior conversation we feed back to the model. Keeps cost bounded
# and avoids confusing the model with stale context after long sessions.
HISTORY_TURN_LIMIT = 6  # = 3 prior question/answer pairs


def _format_rows(rows: list[dict[str, Any]] | None, max_rows: int = 20) -> str:
    if not rows:
        return "(no rows)"
    df = pd.DataFrame(rows)
    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows)
    out = df.to_string(index=False)
    if truncated:
        out += f"\n... ({len(rows) - max_rows} more rows)"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli")
    ap.add_argument("--model", default=os.environ.get("FIREWORKS_MODEL"),
                    help="Fireworks model id (overrides FIREWORKS_MODEL env var)")
    ap.add_argument("--db", default=os.environ.get("CHINOOK_DB", "data/Chinook.db"),
                    help="path to SQLite database")
    args = ap.parse_args(argv)

    try:
        conn = load_db(args.db)
    except FileNotFoundError:
        print(f"error: database not found at '{args.db}'.", file=sys.stderr)
        print("Run ./setup.sh to download Chinook, or pass --db to point at another SQLite file.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: couldn't open database '{args.db}': {e}", file=sys.stderr)
        return 1

    try:
        agent = SqlAgent(conn) if args.model is None else SqlAgent(conn, model=args.model)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: couldn't initialize agent: {e}", file=sys.stderr)
        return 1

    short_model = agent.model.split("/")[-1]
    print(f"text-to-SQL CLI · model={short_model} · db={args.db}")
    print("Ask a natural-language question and get back the SQL plus results.")
    print("Type 'help' for commands, 'exit' to quit.")

    history: list[dict[str, str]] = []  # alternating user/assistant turns

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            return 0
        if q.lower() == "help":
            print(HELP)
            continue
        if q.lower() == "schema":
            try:
                print_table_schema(conn)
            except Exception as e:
                print(f"error: couldn't read schema: {e}")
            continue
        if q.lower() == "reset":
            history.clear()
            print("(conversation reset)")
            continue

        prior = history[-HISTORY_TURN_LIMIT:] if history else None

        t0 = time.perf_counter()
        try:
            result = agent.ask(q, prior_turns=prior, on_status=lambda m: print(f"  · {m}"))
        except Exception as e:
            # Catch-all for anything not handled inside ask() — keeps the REPL alive
            # for the next question instead of dumping a traceback.
            print(f"\nerror: unexpected failure: {e}")
            continue
        dt = (time.perf_counter() - t0) * 1000

        if result.error:
            print(f"\n✗ couldn't answer that — {dt:.0f} ms, {result.attempts} attempt(s)")
            print(f"  {result.error}")
            if result.sql:
                print(f"  last SQL: {result.sql}")
            continue

        repair_note = (
            f" · {result.repairs} repair{'s' if result.repairs != 1 else ''}"
            if result.repairs else ""
        )
        print(f"\nSQL · {dt:.0f} ms{repair_note}")
        print(f"  {result.sql}")
        print()
        print(_format_rows(result.rows))

        # Carry forward a compact representation of this turn for follow-ups.
        history.append({"role": "user", "content": q})
        history.append({
            "role": "assistant",
            "content": json.dumps({"sql": result.sql, "rationale": result.rationale}),
        })

    return 0


if __name__ == "__main__":
    sys.exit(main())
