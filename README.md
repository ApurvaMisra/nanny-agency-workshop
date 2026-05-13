# Nanny Agency — AI Reliability Workshop

A 6-hour hands-on workshop covering one-turn LLM patterns, agent reliability, and domain-aware evaluation, themed around a fictional nanny agency.

## Quick start

```bash
uv sync --all-extras
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
jupyter notebook notebooks/00_smoke_test.ipynb
```

## Structure

- `notebooks/00_smoke_test.ipynb` — verifies your environment before the workshop
- `notebooks/01_one_turn.ipynb` — one-turn LLM patterns (added in Plan 2)
- `notebooks/02_agent.ipynb` — BAML agent + reliability (added in Plan 3)
- `notebooks/03_evaluation.ipynb` — domain-aware evaluation (added in Plan 4)

## Pre-workshop checklist

Run the smoke-test notebook. Every cell must complete with a green checkmark.
