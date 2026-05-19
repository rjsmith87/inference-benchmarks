"""Smoke test for Baseten Model API: streaming + TTFT + json_mode behavior.

Runs ONE question against TWO models:
  - moonshotai/Kimi-K2.6 (json_mode supported)
  - deepseek-ai/DeepSeek-V3.1 (json_mode NOT supported per /v1/models)

Verifies:
  1. Streaming works and we can capture TTFT.
  2. usage tokens are returned in the final chunk when stream_options is set.
  3. A model without json_mode either tolerates response_format silently or errors;
     either way we want to see the failure shape before the full sweep.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load .env manually (avoid dotenv frame quirk same as model_sweep.py).
for ln in Path(".env").read_text().splitlines():
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k, v.strip())

from openai import OpenAI

from src.agent import SYSTEM_PROMPT, _format_schema_compact, _trim_schema_for_question
from src.utils import load_db

BASETEN_BASE_URL = "https://inference.baseten.co/v1"

QUESTION = "Which 5 genres generated the most revenue, and how much did each make?"


def one_call(model: str, supports_json: bool) -> dict:
    api_key = os.environ["BASETEN_API_KEY"]
    client = OpenAI(base_url=BASETEN_BASE_URL, api_key=api_key)
    conn = load_db("data/Chinook.db")
    schema = _trim_schema_for_question(
        conn, QUESTION, _format_schema_compact(conn), compact=True
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": QUESTION},
    ]
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
    )
    if supports_json:
        kwargs["response_format"] = {"type": "json_object"}

    print(f"\n=== {model} (json_mode={'on' if supports_json else 'off'}) ===")
    t0 = time.perf_counter()
    ttft_ms = None
    chunks_seen = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    try:
        stream = client.with_options(timeout=60.0).chat.completions.create(**kwargs)
        for chunk in stream:
            chunks_seen += 1
            if chunk.choices:
                delta = chunk.choices[0].delta
                # TTFT = first chunk with any payload (content or reasoning).
                got_payload = False
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                    got_payload = True
                if getattr(delta, "reasoning_content", None):
                    reasoning_parts.append(delta.reasoning_content)
                    got_payload = True
                if got_payload and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
            if getattr(chunk, "usage", None):
                usage = chunk.usage
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    total_ms = (time.perf_counter() - t0) * 1000
    content = "".join(content_parts)
    parsed_ok = False
    sql = None
    try:
        parsed = json.loads(content)
        parsed_ok = isinstance(parsed, dict) and "sql" in parsed
        if parsed_ok:
            sql = parsed["sql"]
    except json.JSONDecodeError:
        pass

    out = {
        "ok": True,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "chunks": chunks_seen,
        "content_chars": len(content),
        "reasoning_chars": sum(len(p) for p in reasoning_parts),
        "json_parsed": parsed_ok,
        "sql": sql,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "raw_preview": content[:200],
    }
    for k, v in out.items():
        if k == "raw_preview":
            continue
        print(f"  {k}: {v}")
    print(f"  raw_preview: {content[:200]!r}")
    return out


if __name__ == "__main__":
    one_call("moonshotai/Kimi-K2.6", supports_json=True)
    time.sleep(2)
    one_call("deepseek-ai/DeepSeek-V3.1", supports_json=False)
