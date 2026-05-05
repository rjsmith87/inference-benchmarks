"""Text-to-SQL agent.

Architecture:
- Schema rendered into the system prompt (single round trip; no tool-calling overhead).
- Structured JSON output ({"sql": ..., "rationale": ...}) so the response is parseable.
- Execute-and-repair loop: run the SQL; if SQLite raises, send the error back to the
  model and ask for a fix. Up to MAX_REPAIRS retries, then give up.
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from src.utils import query_db

load_dotenv()

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/kimi-k2p6"
MAX_REPAIRS = 2  # 1 initial attempt + up to 2 repairs = 3 model calls max
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 2.0  # seconds; exponential: 2, 4, 8, 16

SYSTEM_PROMPT = """You are an expert SQLite analyst. Convert the user's natural-language \
question into a single valid SQLite SELECT query that answers it.

Output format: a JSON object with these keys, and nothing else:
- "sql": the query, as a single statement, no trailing semicolon.
- "rationale": one short sentence explaining the query.

Rules:
- Use ONLY tables and columns from the schema below.
- Generate exactly ONE SELECT statement. No DDL, no DML, no semicolons inside.
- SQLite dialect (e.g. strftime('%Y', date_col), || for string concat, LIMIT N).
- Quote string literals with single quotes.
- If the question implies a limit ("top 5", "best", "most"), include LIMIT.
- Return ONLY the columns the user asked for. Do NOT add surrogate keys
  (CustomerId, TrackId, EmployeeId, etc.) or extra descriptive columns
  unless the question explicitly requests them.
- When the question asks for a single combined name field (e.g. "the
  employee's name", "the customer's full name"), concatenate FirstName and
  LastName with `||' '||` and alias the result. When the question asks for
  "names" (plural) alongside other fields like emails, return FirstName and
  LastName as separate columns — do NOT concatenate.
- When the question is about "sales" or "revenue", use SUM(UnitPrice * Quantity)
  on InvoiceLine — not raw Quantity counts. Use raw quantities only when the
  question asks for "units", "tracks sold", "count", or similar.

Schema:
{schema}
"""


def _format_schema(conn: sqlite3.Connection) -> str:
    """Render the live schema as the actual CREATE statements stored by SQLite.

    Pulls DDL straight from `sqlite_master` instead of reconstructing it from
    `PRAGMA table_info`. This preserves foreign keys, CHECK constraints,
    DEFAULTs, table options (WITHOUT ROWID), and views — all useful signals for
    the model — and works for any SQLite database without schema-specific code.
    """
    rows = query_db(
        conn,
        "SELECT sql FROM sqlite_master "
        "WHERE type IN ('table', 'view') "
        "  AND name NOT LIKE 'sqlite_%' "
        "  AND sql IS NOT NULL "
        "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name",
        return_as_df=False,
    )
    if not rows:
        return "(no tables in database)"
    return ";\n\n".join(r["sql"].rstrip(";").strip() for r in rows) + ";"


def _format_schema_compact(conn: sqlite3.Connection) -> str:
    """Render the schema as one line per table: ``Table(col:type[pk,fk→Other.Col], ...)``.

    Roughly 62% smaller than the CREATE-TABLE form for Chinook (~390 prompt
    tokens vs ~1040). Latency-driven: most of the per-call latency on the
    shared serverless tier is prefill, so cutting the schema is the highest-
    leverage knob short of moving off shared. Verified on the 10-question dev
    set — no regression vs the verbose form.
    """
    tables = query_db(
        conn,
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        return_as_df=False,
    )
    if not tables:
        return "(no tables in database)"
    lines: list[str] = []
    for t in tables:
        name = t["name"]
        cols = query_db(conn, f"PRAGMA table_info([{name}])", return_as_df=False)
        fks = query_db(conn, f"PRAGMA foreign_key_list([{name}])", return_as_df=False)
        fk_targets = {fk["from"]: f"{fk['table']}.{fk['to']}" for fk in fks}
        col_strs: list[str] = []
        for c in cols:
            tags: list[str] = []
            if c.get("pk"):
                tags.append("pk")
            if c["name"] in fk_targets:
                tags.append(f"fk→{fk_targets[c['name']]}")
            type_short = (c.get("type") or "").split("(")[0].strip().lower()
            piece = c["name"]
            if type_short:
                piece += f":{type_short}"
            if tags:
                piece += f"[{','.join(tags)}]"
            col_strs.append(piece)
        lines.append(f"{name}({', '.join(col_strs)})")
    return "\n".join(lines)


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = query_db(
        conn,
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        return_as_df=False,
    )
    return [r["name"] for r in rows]


def _trim_schema_for_question(
    conn: sqlite3.Connection, question: str, full_schema: str, compact: bool = False
) -> str:
    """Return only the schema for tables likely relevant to the question.

    Matching strategy:
      1. Keyword match table names (singular and plural form) against the question.
      2. Expand the matched set with any table reachable via foreign key.
      3. If fewer than 2 tables match, fall back to the full schema (the keyword
         match is too uncertain to risk a missing-table repair round trip).

    When ``compact`` is True, emits the one-line-per-table compact form
    (matching ``_format_schema_compact``); otherwise the original CREATE TABLE
    DDL form. The trim and the format are independent knobs.
    """
    tables = _list_tables(conn)
    if not tables:
        return full_schema

    qlower = question.lower()
    matched: set[str] = set()
    for t in tables:
        candidates = {t.lower()}
        # Singular ↔ plural fold
        if t.lower().endswith("s"):
            candidates.add(t.lower()[:-1])
        else:
            candidates.add(t.lower() + "s")
        for cand in candidates:
            if re.search(rf"\b{re.escape(cand)}\b", qlower):
                matched.add(t)
                break

    if len(matched) < 2:
        return full_schema

    # Expand via FK to keep JOINs valid.
    expanded = set(matched)
    for t in list(matched):
        try:
            fks = query_db(
                conn, f"PRAGMA foreign_key_list({t})", return_as_df=False
            )
            for fk in fks:
                if fk.get("table"):
                    expanded.add(fk["table"])
        except Exception:
            pass

    if compact:
        # Re-render only the expanded tables in compact form.
        lines: list[str] = []
        for name in sorted(expanded):
            cols = query_db(conn, f"PRAGMA table_info([{name}])", return_as_df=False)
            fks = query_db(conn, f"PRAGMA foreign_key_list([{name}])", return_as_df=False)
            fk_targets = {fk["from"]: f"{fk['table']}.{fk['to']}" for fk in fks}
            col_strs: list[str] = []
            for c in cols:
                tags: list[str] = []
                if c.get("pk"):
                    tags.append("pk")
                if c["name"] in fk_targets:
                    tags.append(f"fk→{fk_targets[c['name']]}")
                type_short = (c.get("type") or "").split("(")[0].strip().lower()
                piece = c["name"]
                if type_short:
                    piece += f":{type_short}"
                if tags:
                    piece += f"[{','.join(tags)}]"
                col_strs.append(piece)
            lines.append(f"{name}({', '.join(col_strs)})")
        return "\n".join(lines) if lines else full_schema

    placeholders = ",".join("?" * len(expanded))
    ddl = query_db(
        conn,
        f"SELECT sql FROM sqlite_master "
        f"WHERE type='table' AND name IN ({placeholders}) AND sql IS NOT NULL "
        f"ORDER BY name",
        params=tuple(expanded),
        return_as_df=False,
    )
    if not ddl:
        return full_schema
    return ";\n\n".join(r["sql"].rstrip(";").strip() for r in ddl) + ";"


@dataclass
class AgentResult:
    question: str
    sql: str | None
    rows: list[dict[str, Any]] | None
    error: str | None
    attempts: int                       # number of model calls made
    repairs: int                        # number of repair turns used (0..MAX_REPAIRS)
    rationale: str | None = None
    prompt_tokens: int = 0              # summed across attempts
    completion_tokens: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)


class SqlAgent:
    def __init__(
        self,
        conn: sqlite3.Connection,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_repairs: int = MAX_REPAIRS,
        extra_body: dict[str, Any] | None = None,
        trim_schema: bool = True,
        compact_schema: bool = True,
    ):
        self.conn = conn
        self.model = model
        self.max_repairs = max_repairs
        self.trim_schema = trim_schema
        self.compact_schema = compact_schema
        # Qwen3 emits a <think> block by default; reasoning_effort=none suppresses it.
        self.extra_body = extra_body or ({"reasoning_effort": "none"} if "qwen3" in model else {})
        key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. Put it in .env or export it."
            )
        self.client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=key)
        # Compact schema cuts ~62% of schema chars (~390 vs ~1040 prompt tokens),
        # which matters for shared-tier prefill latency. The verbose CREATE
        # form stays available via compact_schema=False if a customer schema
        # has hints (CHECK constraints, DEFAULTs) only the verbose form carries.
        self.full_schema = (
            _format_schema_compact(conn) if compact_schema else _format_schema(conn)
        )
        self.full_schema_verbose = _format_schema(conn)
        # Backwards-compat alias for any caller that read this directly.
        self.schema_str = self.full_schema

    def _call(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int, int]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        # Retry with exponential backoff + jitter on 429s. Fireworks sometimes
        # caps small-model deployments at low per-minute QPS; surfacing a hard
        # error to the user is worse than waiting a few seconds.
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                if attempt >= RATE_LIMIT_RETRIES:
                    raise
                delay = RATE_LIMIT_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

        content = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        return json.loads(content), pt, ct

    def ask(
        self,
        question: str,
        prior_turns: list[dict[str, str]] | None = None,
        on_status: "callable | None" = None,
    ) -> AgentResult:
        # Trim the schema to tables likely relevant to the question; fall back
        # to the full schema if the keyword matcher is uncertain. On a repair
        # turn we swap to the full (compact) schema (the trim may have missed
        # something). The schema *form* (compact vs verbose) is independent of
        # whether trim fired.
        if self.trim_schema:
            initial_schema = _trim_schema_for_question(
                self.conn, question, self.full_schema, compact=self.compact_schema
            )
        else:
            initial_schema = self.full_schema

        def system_msg(schema: str) -> dict[str, str]:
            return {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)}

        messages: list[dict[str, str]] = [system_msg(initial_schema)]
        if prior_turns:
            messages.extend(prior_turns)
        messages.append({"role": "user", "content": question})

        last_sql: str | None = None
        last_rationale: str | None = None
        last_error: str | None = None
        total_pt = 0
        total_ct = 0

        for repair in range(self.max_repairs + 1):
            try:
                parsed, pt, ct = self._call(messages)
            except Exception as e:
                return AgentResult(
                    question=question, sql=last_sql, rows=None,
                    error=f"model call failed: {e}",
                    attempts=repair + 1, repairs=repair,
                    prompt_tokens=total_pt, completion_tokens=total_ct,
                    messages=messages,
                )

            total_pt += pt
            total_ct += ct
            sql = (parsed.get("sql") or "").strip().rstrip(";").strip()
            rationale = parsed.get("rationale")
            last_sql = sql or last_sql
            last_rationale = rationale or last_rationale
            messages.append({"role": "assistant", "content": json.dumps(parsed)})

            if not sql:
                last_error = "model returned empty SQL"
            else:
                try:
                    rows = query_db(self.conn, sql, return_as_df=False)
                    return AgentResult(
                        question=question, sql=sql, rows=rows,
                        error=None, attempts=repair + 1, repairs=repair,
                        rationale=rationale,
                        prompt_tokens=total_pt, completion_tokens=total_ct,
                        messages=messages,
                    )
                except sqlite3.Error as e:
                    last_error = str(e)

            if repair >= self.max_repairs:
                break

            if on_status:
                on_status(f"SQL failed — retrying ({repair + 1}/{self.max_repairs})…")

            # On the first repair, swap in the full schema in case the trim
            # was too aggressive and dropped a needed table.
            if repair == 0 and self.trim_schema and initial_schema is not self.full_schema:
                messages[0] = system_msg(self.full_schema)

            messages.append({
                "role": "user",
                "content": (
                    f"Your query failed: {last_error}\n"
                    "Return a corrected query in the same JSON format."
                ),
            })

        return AgentResult(
            question=question, sql=last_sql, rows=None,
            error=last_error, attempts=self.max_repairs + 1,
            repairs=self.max_repairs, rationale=last_rationale,
            prompt_tokens=total_pt, completion_tokens=total_ct,
            messages=messages,
        )
