# Fireworks Text-to-SQL — Submission

An interactive CLI that converts natural-language questions into executable
SQLite queries against the Chinook music-store database, using open-source
models hosted on Fireworks.

## Quick start

```bash
./setup.sh                            # uv venv, deps, downloads Chinook.db
source .venv/bin/activate
echo 'FIREWORKS_API_KEY=fw_...' > .env

# Interactive CLI
python -m src.cli

# Run the 10 dev questions and write the required answer file
python -m src.evals accounts/fireworks/models/kimi-k2p6 \
    --out-answers data/dev_answers.json

# Run the 15-question synthetic pattern-coverage set
python -m src.evals accounts/fireworks/models/kimi-k2p6 \
    --questions data/synthetic_questions.json

# Replay the customer's naive prompt as a baseline (writes data/baseline_results.json)
python -m src.evals accounts/fireworks/models/kimi-k2p6 --baseline

# (Re)build the synthetic set from gold SQL run live against Chinook
PYTHONPATH=. python scripts/build_synthetic.py

# Latency + cost benchmarking against the customer's volume profile
python -m src.perf accounts/fireworks/models/kimi-k2p6 --out perf_kimi.json

# Single-page interactive dashboard (problem, results, live demo against
# Fireworks + sql.js executing queries in your browser)
python3 -m http.server 8000          # then open http://localhost:8000/dashboard.html
```

### Environment

- `FIREWORKS_API_KEY` (required) — picked up from `.env` via `python-dotenv`
  or from the process environment.
- `FIREWORKS_MODEL` (optional) — overrides the default model for the CLI.
- `CHINOOK_DB` (optional) — overrides the database path (default
  `data/Chinook.db`).

Python 3.11+. The provided `setup.sh` uses `uv`; if `uv` is unavailable,
`python3 -m venv .venv && pip install -e .` works equivalently.

## What's where

```
src/
  agent.py        # SqlAgent — schema-in-prompt, JSON output, execute-and-repair
  cli.py          # interactive REPL with follow-up support
  evals.py        # tolerant eval framework + dev_answers.json writer
  perf.py         # latency + cost benchmarking with token-usage capture
  utils.py        # provided DB helpers (unmodified)
scripts/
  build_synthetic.py   # generates data/synthetic_questions.json from gold SQL
data/
  Chinook.db
  dev_questions_with_answers.json   # provided gold set (10 questions)
  dev_answers.json                  # this submission's outputs
  synthetic_questions.json          # 15 hand-written pattern-coverage questions
dashboard.html                      # single-page submission summary + live demo
perf_kimi.json                      # latest perf snapshot
```

## Architecture

A single-shot agent with an execute-and-repair loop, intentionally simple:

1. **Schema in the system prompt.** The full Chinook schema is rendered as
   `CREATE TABLE` blocks and prepended to every request. With 11 small tables
   this fits in ~900 prompt tokens and avoids a tool-calling round trip.
2. **Structured JSON output.** `response_format={"type": "json_object"}` plus
   an explicit "{sql, rationale}" contract in the system prompt. This keeps
   parsing trivial and gives a one-line rationale for human review.
3. **Execute-and-repair, max 2 retries.** If SQLite raises on the generated
   query, the agent feeds the error back as a follow-up turn and asks the
   model for a corrected version. Up to two repair turns before giving up.
4. **429 retry with exponential backoff** on Fireworks rate limits. Smaller-
   model deployments throttle more aggressively; surfacing a hard error to
   the user is worse than waiting a few seconds.
5. **Conversation history for follow-ups.** The CLI carries the last six
   turns into each subsequent call, so questions like *"Now show the tracks
   on the first one"* resolve correctly against prior answers.

### Default model

`accounts/fireworks/models/kimi-k2p6` (Kimi K2.6).

The customer asked specifically for Qwen2.5-Coder-32B-Instruct, but it was
not reachable from the supplied Fireworks key. Kimi K2.6 was selected
after a catalog-wide sweep against the 10 dev questions
(`scripts/model_sweep.py`, results in `data/model_sweep.json`). It was the
only chat-capable model on this key that combined high JSON-mode
reliability, sub-3s clean p50, and (with the repair loop) 10/10 dev
accuracy. The model is a constructor arg on `SqlAgent` and a CLI flag
(`--model`), so swapping it is one line.

### Catalog sweep — every chat-capable model on this key

`scripts/model_sweep.py` runs the 10 dev questions against each model in a
single-shot, no-repair, JSON-mode-required configuration so the comparison
isolates raw model behavior. Pacing 1.5s/call, 30s per-call timeout. Run:

```bash
PYTHONPATH=. python scripts/model_sweep.py --out data/model_sweep.json
```

| Model | Raw | Clean (excl. 429s) | JSON honored | p50 (clean) | Notes |
|---|---|---|---|---|---|
| **Kimi K2.6** (primary) | 6/10 | 6/7 | 7/10 | 1,294 ms | Ships in this POC. With the agent's repair + retry, hits 10/10 reliably. |
| Kimi K2.5 | 7/10 | 7/7 | 7/10 | 7,917 ms | Best raw single-shot accuracy, but ~5× slower than K2.6. Drop-in alternative. |
| DeepSeek V4 Pro | 4/10 | 4/6 | 6/10 | 2,956 ms | Reasoning-class output (~282 mean completion tokens vs Kimi's ~48). 1 call returned empty content. |
| GLM-5 | 3/10 | 3/3 | 3/10 | 5,584 ms | 7/10 calls hit 429s. Of the 3 clean calls, all 3 were correct, but JSON mode mostly ignored. |
| GLM-5.1 | 2/10 | 2/3 | 3/10 | 9,817 ms | Same 7/10 429s as GLM-5. JSON unreliable. One SQLite-rejected query. |
| Minimax M2.7 | 3/10 | 3/3 | 3/10 | 3,262 ms | Same 7/10 429s. Of clean calls, all correct. Worth re-testing on a fresh rate-limit window. |
| Qwen3-8B (legacy) | 5/10 | — | — | ~3-5 s | Tested in an earlier session; not currently visible to this key. CACR-router candidate. |
| Qwen2.5-Coder-32B | — | — | — | — | Customer ask · not reachable from supplied key. |

The 429 rate-limit hits during this sweep make the GLM and Minimax
numbers noisy — small denominators in the "clean" column. The choice to
ship Kimi K2.6 is reinforced by its repair-loop accuracy (10/10) and the
fact that pricing is verified for it specifically. Re-running the sweep
on priority access would let us confirm or rule out the GLM/Minimax tier
as cheap-first router candidates.

## Results

### Customer baseline (`evals.py --baseline`)

The customer's current prototype prompt (`Convert this question to SQL:
{question}` — no schema, no JSON mode, no repair) on the same Kimi K2.6
endpoint scored **1/10** on the dev set. The single pass — q_004,
"most popular media type" — survived only because the model happened to
guess the correct table casing (`MediaType`, `Track`). Every other question
referenced lowercase plural names (`tracks`, `customers`, `albums`) that
don't exist in Chinook. Full enriched output is in `data/baseline_results.json`.

That number is the lift baseline: schema injection, JSON mode, and the
execute-and-repair loop together turn a **1/10 system into 10/10 on the
same model and same key.**

### Accuracy on the 10 dev questions (`evals.py`)

| Model | Tier 1 | Tier 2 | Tier 3 | Total |
|---|---|---|---|---|
| **Kimi K2.6 + agent** (primary) | 4/4 | 4/4 | 2/2 | **10/10 (100%)** |
| Kimi K2.6 + naive baseline prompt | 1/4 | 0/4 | 0/2 | 1/10 (10%) |
| Qwen3-8B + agent (cheap candidate) | 2/4 | 3/4 | 0/2 | 5/10 (50%) |

### Accuracy on the 15-question synthetic pattern-coverage set

The dev set, while useful, doesn't exercise common SQL patterns like
subqueries, `CASE`, `COALESCE`, `UNION`, self-joins, `IS NULL`, date
arithmetic, `LIKE`, or top-per-group. `data/synthetic_questions.json`
fills that gap with 15 hand-written questions whose gold SQL is run
against Chinook live (in `scripts/build_synthetic.py`) so the
expected results are always consistent with the database.

| Model | Tier 1 | Tier 2 | Tier 3 | Total |
|---|---|---|---|---|
| **Kimi K2.6** | 4/4 | 8/8 | 1/3 | **13/15 (87%)** |

Patterns covered:

```
subquery_where         CASE in SELECT       IS NULL filter
date range filter      self-join             COUNT(DISTINCT)
UNION                  nested aggregation    COALESCE
LIKE wildcard          correlated subquery   CASE in GROUP BY
date arithmetic ←  flaky    LIKE + HAVING    top-per-group  ←  hard miss
```

The two failures:

- **s_014 (top-per-group)** is a hard miss. *"Who is the top-spending
  customer for each support rep?"* The model interprets "for each
  support rep" as a column selector rather than a partitioning
  constraint — generates `GROUP BY (rep, customer) ORDER BY spending
  DESC` and returns all 59 rep-customer pairs instead of the one top
  per rep. The dev set's q_009 uses a window function (`RANK()`) that
  Kimi handles correctly, but q_009 asks for a global top-5; s_014
  needs `ROW_NUMBER() OVER (PARTITION BY rep ORDER BY spending DESC)
  WHERE rn = 1`, a stricter pattern. One or two few-shot examples
  would likely close it; an RFT loop with executor feedback would
  close it permanently.
- **s_013 (date arithmetic)** is run-to-run flaky at temperature 0.
  *"Total revenue in the last 6 months of available invoice data"* is
  ambiguous, and Kimi sometimes generates the gold pattern
  (`date(MAX(InvoiceDate),'-6 months')`) and sometimes anchors the
  window differently. Counted as a failure to be conservative; the
  dashboard re-runs this card live so the variance is visible.

### Combined: 23/25 (92%) across both eval sets

| | Dev set (10) | Synthetic set (15) | Combined |
|---|---|---|---|
| Tier 1 | 4/4 | 4/4 | 8/8 |
| Tier 2 | 4/4 | 8/8 | 12/12 |
| Tier 3 | 2/2 | 1/3 | 3/5 |
| **Total** | **10/10** | **13/15** | **23/25 (92%)** |

Comparison uses a tolerant multiset comparator (column-rename safe,
float-epsilon, order-insensitive) — standard convention in text-to-SQL
benchmarks (Spider, BIRD).

### Latency

Latency is Raul's hardest constraint — sub-3 s P50 end-to-end. The
honest answer: **shared serverless gets us under 3 s some of the time,
but not contractually.** Three back-to-back `perf.py` runs against the
10-question dev set with the compact schema:

| Run | p50 | p90 | accuracy | note |
|---|---|---|---|---|
| 1 | 9,067 ms | 22,065 ms | 10/10 | shared tier under load |
| 2 | 8,297 ms | 39,673 ms | 6/10 | three queries hit 429s and never recovered |
| 3 | **4,058 ms** | 17,935 ms | 10/10 | quieter shared tier · cleanest run |

Across all 30 calls: median 6,276 ms, **23% under 3 s, 33% under 5 s,
53% under 10 s.** The dev eval (no inter-query pacing) hit a clean
median of 2,901 ms separately — the difference between perf.py (paced 1
s) and evals.py (no pacing) is run-to-run load variance, not pacing.

What we did to push down latency:

- **Compact schema** is the highest-leverage knob. Rendering the schema
  as one line per table (`Album(AlbumId:integer[pk], Title:nvarchar,
  ArtistId:integer[fk→Artist.ArtistId])`) instead of full
  `CREATE TABLE` blocks dropped mean prompt tokens from **935 → 665
  (29% reduction)** with no accuracy regression on the dev set. See
  `_format_schema_compact` in `src/agent.py`.
- **Schema trim** (keyword + FK-closure) fires on 9/25 questions; mean
  reduction across all 25 is **24.7%**, mean reduction on the 9 fired
  is **68.5%**. On the worst case the trim drops 89.8% of schema
  characters (q_002 — only Album/Artist needed).
- **`service_tier="priority"` is silently accepted but does nothing.**
  Fireworks priority is account-provisioned, not a request-time flag.
  We tried both `extra_body={"service_tier":"priority"}` and the
  `X-Fireworks-Server-Type` header — neither produced a measurable
  latency change. Path to a firm SLO is platform configuration
  (on-demand or dedicated deployment), not code.

### Cost (`perf.py`, Kimi K2.6, 10-question dev set, compact schema)

| Metric | Value |
|---|---|
| Accuracy | 100% (10/10) on clean runs |
| Tokens per query (mean) | ~665 prompt, ~78 completion |
| Cost per query | **$0.000942** (Fireworks Kimi K2.6: $0.95/$4.00 per 1M) |
| Daily cost @ 30k q/day | $28.26 |
| Monthly cost @ 30k q/day | $848 |
| GPT-5.4 baseline at same volume | $84.77/day · $2,543/mo (**~67% savings**) |

The 30k q/day volume is from Raul's email (1,000 users × 30 queries/day).
GPT-5.4 baseline rates are the customer-cited $2.50 input / $15.00 output
per 1M tokens; we apply them to the same token-count profile our agent
produced, which is conservative for the baseline (a less-tightly-prompted
proprietary baseline would likely use more tokens per query, not fewer).
The compact-schema rollout cut another ~21% off our per-query cost vs the
verbose schema baseline ($0.001197 → $0.000942).

### Dashboard

`dashboard.html` is a single-file interactive submission summary. It
loads sql.js + Chinook.db in the browser and lets the reviewer:

- Re-run any of the 25 questions live against Fireworks (their own key)
  and see SQL, results, pass/fail vs gold answers.
- Run the customer's naive baseline prompt and our agent prompt
  side-by-side on the same question — watch 1/10 vs 10/10 with their
  own eyes.
- Run the same question on Kimi K2.6 vs Qwen3-8B side-by-side.
- See the live latency distribution (loaded from
  `perf_compact_*.json`) with the 3-second line marked.
- Drag a slider to project cost at 1k–100k q/day vs GPT-5.4.
- Click any architecture node or design decision card for the
  considered-alternatives + tradeoff.

The API key is held in a password input and used only for direct
requests to api.fireworks.ai. Never persisted.

## Known limitations

1. **Sub-3 s P50 on shared serverless is not contractual.** Across 30
   calls in 3 perf runs we saw 23% under 3 s, 33% under 5 s, 53% under 10 s.
   Best clean p50 was 4,058 ms (run 3); a separate dev-eval pass hit
   median 2,901 ms with no pacing. The variance is load-side, not a
   model property. A publish-quality SLO requires on-demand or dedicated
   deployment — see the roadmap. Do not promise the customer 3 s P50 on
   shared serverless.
2. **`service_tier="priority"` is silently accepted but does nothing.**
   Fireworks priority is account-provisioned, not a request-time flag.
3. **Schema trim is keyword-based, not retrieval.** Fires on 9 of 25
   questions today (mean 68.5% reduction on those, 24.7% across all 25).
   For a 1,000-table customer DB this isn't enough; we'd need a real
   retrieval step (embed the question, find relevant tables) or schema
   caching with a stable cache id.
4. **`format_answer_summary` is heuristic.** Renders rows as
   `<entity name> (col=val, col=val)` with a small carve-out for the
   FirstName+LastName pattern. The values are exact; the wording may not
   match the gold set's phrasing verbatim. Eval correctness is judged by
   `rows_match` against the SQL result, not by string-comparison of the
   `answer` field.
5. **s_013 (date-arithmetic) is run-to-run flaky at temperature 0.**
   The "last 6 months of available data" pattern is genuinely ambiguous;
   we count it as a failure to be conservative. The dashboard re-runs
   this card live so the variance is visible.
6. **Qwen3-8B's 50% should be read with care.** Some failures (q_001:
   units vs revenue) are real semantic misses, but others would likely
   improve with one or two few-shot examples. The 8B model is plausible as
   a "fast first attempt" inside a CACR-style router, not as a solo
   replacement.
7. **Pricing for Qwen3-8B is still an estimate** in `perf.py`'s
   `PRICING_USD_PER_M` dict; verify before publishing any cheap-model cost
   number.
8. **Rate limits hit during perf run 2.** Three of ten queries got 429
   responses and the eval counted them as wrong. The 60% accuracy on that
   run is not a quality regression; it's the shared-tier per-minute QPS
   cap. Production would need either pacing, retries beyond our 4-step
   backoff, or a higher rate-limit allowance.

## Things I'd do next, in order

1. **Fine-tune a small open-source model on Chinook-style schemas.**
   Generate ~1k synthetic (question, gold_sql) pairs across diverse
   schemas, then SFT a Qwen3-class model. The biggest single wins for
   text-to-SQL come from teaching the model the *output discipline* (no
   surrogate IDs, correct GROUP BY structure, idiomatic LIMIT/RANK use) —
   exactly the patterns the system prompt is bandaging today. SFT first to
   establish the format, then DPO with executor-feedback pairs (working
   query preferred over failing query) to close the long tail.
2. **Schema retrieval for production.** Replace the full schema dump with
   a tables-and-columns shortlisting step: embed the question, retrieve
   the top-k relevant tables, render only those. Cuts per-query cost and
   makes the architecture survive a 1000-table customer database.
3. **CACR-style router** (cheap-first, escalate on low confidence). The
   takehome already has the building blocks: Qwen3-8B handles the easy
   tier-1s for fractions of a cent, Kimi catches the rest. Confidence
   probes can come from logprobs (cheaper than a second LLM call).
4. **Multi-run perf measurement** with separate model-time and
   wall-time tracking, so customer-facing numbers aren't contaminated by
   429 backoff sleeps.
5. **End-to-end eval expansion.** 25 questions across two sets is
   enough to characterize behavior on common patterns and catch a
   real failure mode (top-per-group). A 200-question set with tier
   balance, intentional ambiguity, and explicit pattern tagging
   (continuing from `synthetic_questions.json`'s `pattern` field)
   would let us measure prompt and model changes with statistical
   confidence and track per-pattern accuracy over time — the key
   signal for prioritizing fine-tune training data.

---

This submission was developed with AI coding assistance (Claude). The
agent code, eval framework, perf instrumentation, and CLI were all written
collaboratively, then verified end-to-end against the dev set.
