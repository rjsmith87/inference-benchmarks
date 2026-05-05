# Email to Raul Jimenez

**From:** Robert Smith
**To:** Raul Jimenez <raul.j@gitlab.com>
**Subject:** Re: Help Needed: Agentic BI CLI Product for GitLab Customers

Hi Raul,

I built the POC you described. Before reading the rest of this email, open this:

https://rjsmith87.github.io/fireworks-takehome/dashboard.html

Paste a Fireworks API key in the Live Demo tab, pick any two models, pick any question, and click Run. You'll see SQL generated, results executed in your browser, latency measured, and cost projected at your volume. Every number is from real API calls. Nothing is mocked.

**What it does**

An interactive CLI (python -m src.cli) that auto discovers the database schema, generates SQL via open source models on Fireworks, executes it, and repairs on failure. Supports follow up questions.

**Quality**

I evaluated against 25 questions: your 10 plus 15 I wrote to cover gaps (subqueries, self joins, date filtering, NULLs, correlated subqueries). Results on Kimi K2.6:

- Combined: 23/25 (92%)
- Your current baseline prompt with no schema: 2/10
- Two known misses remain (top per group, date window anchoring). Both are addressable with fine tuning.

**Latency**

Your target: under 3s P50. On shared serverless I measured median latency between 900ms and 2,900ms depending on load. Schema compaction (29% token reduction) helps. The variance is infrastructure side. Path to a firm SLO: Fireworks on demand deployment plus prompt caching.

**Cost**

At 30,000 queries/day:

- Fireworks (Kimi K2.6): ~$848/month
- GPT 5.4 at the same token profile: ~$2,549/month
- Savings: ~67%

Total Fireworks spend during development (hundreds of calls across 7 models): under $0.30.

**POC to Production**

Phase 1: Fine tune with SFT for output format, then RFT using the SQL executor as the reward signal. SQL correctness is automatically verifiable so no human labeling is needed. Fireworks supports both.
Phase 2: On demand deployment for predictable sub 3s latency plus prompt caching.
Phase 3: Scale with the second cohort. Fireworks auto scales without infrastructure changes on your side.

**AI disclosure**

I used Claude (Anthropic) throughout. Claude Code wrote the implementation. I drove architecture, model selection, eval design, and this communication.

Happy to walk through any of this on a call.

Best,
Robert Smith
