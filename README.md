# Nanny Agency — AI Reliability Workshop

A 6-hour hands-on workshop covering one-turn LLM patterns, agent reliability with BAML, and domain-aware evaluation. All examples are themed around a fictional **nanny agency** matching nannies to parents.

## What you'll build

1. **Notebook 1** — One-turn LLM patterns: email generation, personalization, PDF extraction, embeddings + matching (ChromaDB).
2. **Notebook 2** — Booking agent in BAML: ReAct loop, multi-tool, progressive disclosure, context management, memory, permissions, fallbacks, determinism, multi-agent with the 5 failure modes, guardrails. Traced with Arize Phoenix.
3. **Notebook 3** — Evaluation: vendor-metric critique, error analysis (open + axial coding), transition failure matrix, synthetic data generation, golden dataset, programmatic + LLM-as-judge (DSPy-optimized), CI integration, online monitoring.

## Pre-workshop setup (do this before the workshop)

### 1. Requirements
- Python 3.11 or newer
- `uv` (recommended) or pip
- An OpenAI API key with a small usage budget (estimated < $5 per attendee, mostly under $1)

### 2. Install

```bash
git clone <repo-url> nanny-agency-workshop
cd nanny-agency-workshop
uv sync --all-extras
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

### 3. Configure your API key

```bash
cp .env.example .env
# Open .env in any editor and replace sk-replace-with-your-key
```

### 4. Generate the BAML client (one-time)

```bash
uv run baml-cli generate --from baml_src
```

This writes `baml_client/` at the repo root. It's gitignored — regenerate after pulling BAML schema changes.

### 5. Run the smoke test

```bash
uv run jupyter notebook notebooks/00_smoke_test.ipynb
```

Run every cell. Each cell should print `✅ OK`. If Phoenix prints `⚠️`, the workshop will fall back to a JSON trace logger — that's fine.

## Troubleshooting

- **`baml-cli generate` fails** — make sure you ran `uv sync --all-extras`; `baml-py` brings the CLI. On Apple Silicon, ensure you're on Python 3.11+ (older Pythons fail to install `baml-py`).
- **Phoenix port already in use** — the smoke test will log a warning; the workshop ships a JSON trace logger fallback.
- **OpenAI 429 (rate limit)** — the workshop uses `gpt-4o-mini` (very low quota). If you hit limits, the on-disk cache means re-runs are free. Wait 60s and re-run the failing cell.
- **PDF extraction returns empty text** — some PDFs have no embedded text. The workshop's PDFs are all text-based; if you see this, the file may have been corrupted on download — `git checkout data/pdfs/` to restore.
- **`baml_client` import error** — run `uv run baml-cli generate --from baml_src` (step 4 above). The generated client is gitignored.

## Repo layout

```
notebooks/                 — workshop notebooks (00 smoke test + 01-03 added in later plans)
data/                      — PDFs, policies, seed DB
src/nanny_workshop/        — shared helpers
baml_src/                  — BAML schemas + agent functions
scripts/                   — one-shot data generators (already run; outputs committed)
tests/                     — pytest suite for helpers + nbmake smoke check
```
