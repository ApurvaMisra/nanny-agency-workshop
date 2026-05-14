# Workshop Setup — Attendee Guide

This is a 6-hour, hands-on workshop on building **reliable LLM applications**. You will write Python in Jupyter notebooks, call the OpenAI API, build agents with BAML, and run a domain-aware evaluation pipeline. Every cell is pre-written — your job is to run, observe, and tweak.

Please complete this setup **before the workshop begins**. The setup takes ~15 minutes including downloads. The smoke-test notebook at the end confirms everything works.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Configure your OpenAI API key](#3-configure-your-openai-api-key)
4. [Generate the BAML client](#4-generate-the-baml-client)
5. [Run the smoke test](#5-run-the-smoke-test)
6. [Tech stack — what each library does](#6-tech-stack--what-each-library-does)
7. [Workshop agenda](#7-workshop-agenda)
8. [Troubleshooting](#8-troubleshooting)
9. [What to bring on the day](#9-what-to-bring-on-the-day)

---

## 1. Prerequisites

| Requirement | Why | Notes |
|---|---|---|
| **Python 3.11 or newer** | Several deps (`baml-py`, modern type syntax) require it | `python3 --version` |
| **`uv`** (recommended) or `pip` | Dependency / venv manager — much faster than pip | Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Git** | To clone the repo | `git --version` |
| **An OpenAI API key with usage budget** | All LLM calls go through OpenAI | Estimated cost: **< $5 total for the full workshop**, usually under $1 thanks to on-disk caching. |
| **A modern browser** | Jupyter UI + Phoenix observability + the Mermaid diagrams in `docs/agent-flow.html` | Chrome, Firefox, Safari, or Edge |
| **macOS, Linux, or WSL2 on Windows** | Tested on macOS 14+ and Linux. Native Windows is not supported. | Apple Silicon and Intel both work |
| **~2 GB free disk** | Python deps + Chroma DB + cached LLM responses | |

You do **not** need a GPU. Everything runs against the OpenAI API.

### Getting an OpenAI API key

1. Sign up at https://platform.openai.com
2. Add a small payment method (you can cap monthly spend at $5)
3. Create a key at https://platform.openai.com/api-keys — copy it (you'll only see it once)

---

## 2. Installation

```bash
# 1. Clone the repo
git clone https://github.com/ApurvaMisra/nanny-agency-workshop.git
cd nanny-agency-workshop

# 2. Install everything (Python 3.11+ + all dependencies)
uv sync --all-extras
```

If you don't have `uv`:

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows-WSL: source .venv/bin/activate
pip install -e ".[dev]"
```

`uv sync` is preferred — it uses `uv.lock` for **byte-for-byte reproducible** installs.

---

## 3. Configure your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` in any editor and replace the placeholder:

```env
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

> **⚠️ Security:** `.env` is gitignored — it will never be committed. Do not paste your key into any chat, screenshot, or notebook cell.

---

## 4. Generate the BAML client

BAML is a typed prompt language — it compiles `.baml` files into a Python client. This step is one-time after install (rerun only if the `.baml` files change).

```bash
uv run baml-cli generate --from baml_src
```

You should see something like `[BAML INFO] Wrote 14 files to baml_client`. A new `baml_client/` directory appears at the repo root (gitignored — regenerate any time).

---

## 5. Run the smoke test

```bash
uv run jupyter notebook notebooks/00_smoke_test.ipynb
```

This opens the smoke-test notebook in your browser. Run every cell (`Shift-Enter`). Each cell should print **`✅ OK`**. If Phoenix prints `⚠️`, that's still fine — the workshop has a graceful fallback.

If every cell printed ✅ (or Phoenix ⚠️), **you're ready for the workshop**. 🎉

---

## 6. Tech stack — what each library does

The whole stack is **Python + open-source + OpenAI**. No proprietary SaaS dependencies beyond the OpenAI API itself.

### Core LLM stack

| Library | Role in the workshop |
|---|---|
| **`openai`** | OpenAI Python SDK — used in Notebook 1 for completions + embeddings. |
| **`baml-py`** + BAML CLI | Schema-first prompt language for the agent in Notebook 2. Compiles `.baml` files into typed Python functions (`b.DecideOneTool(...)`, `b.PlanAgent(...)`, etc.). Gives you deterministic structured outputs without writing regex parsers. |
| **`pydantic`** | Validates every structured LLM output (extraction schemas, agent decisions). When the model invents a field, Pydantic catches it. |
| **`dspy-ai`** | Used in Notebook 3 to **optimize an LLM-as-judge prompt** via `BootstrapFewShot` against human-labeled examples. |

### Retrieval & embeddings

| Library | Role |
|---|---|
| **`chromadb`** | Local persistent vector store (`nanny_db/`). Holds 10 profile embeddings, queryable by metadata. No signup, no API keys. |
| **`umap-learn`** | Projects 1536-dim embeddings to 2D for the interactive scatter in Notebook 1 Section 4. |
| **`scikit-learn`** | Cosine distances and other small-data utilities. |

### Visualization & UI

| Library | Role |
|---|---|
| **`plotly`** + **`pandas`** | Interactive embedding scatter in Notebook 1 — hover any point to see the full profile document and its top-3 nearest neighbors with cosine distances. |
| **`matplotlib`** | Legacy / static plots. |

### Observability & evaluation

| Library | Role |
|---|---|
| **`arize-phoenix`** | Local in-process observability. Auto-traces every OpenAI and BAML call. Open the URL it prints to inspect any conversation's full trace tree (latency, tokens, structured decisions). |
| **`nbmake`** (dev only) | Runs each notebook end-to-end in CI to catch regressions. |
| **`pytest`** (dev only) | Tests the `nanny_workshop` helper library + the eval pipeline. |

### I/O

| Library | Role |
|---|---|
| **`pypdf`** | Text extraction from the 10 nanny resume / parent intake PDFs in Notebook 1. |
| **`reportlab`** | Used **once, offline** to generate the synthetic PDFs you'll find in `data/pdfs/`. Attendees don't touch this. |
| **`python-dotenv`** | Loads `.env` into environment variables. |

### Project tools

| Tool | Role |
|---|---|
| **`uv`** | Python env + dependency manager. Locks via `uv.lock`. |
| **`hatchling`** | Build backend declared in `pyproject.toml`. |
| **`jupyter`** + **`ipykernel`** | The notebook runtime. |

---

## 7. Workshop agenda

| Time | Notebook | What you build |
|---|---|---|
| ~75 min | **`01_one_turn.ipynb`** | One-turn LLM patterns: prompted email generation, personalization, PDF → typed records, vector matching with ChromaDB + interactive UMAP scatter. |
| ~150 min | **`02_agent.ipynb`** | A parent-facing booking agent in BAML. Each section adds one reliability feature in response to a demonstrated failure: ReAct, progressive disclosure, context management, memory, permissions, determinism + fallbacks, multi-agent topology, the 5 multi-agent failure modes, guardrails. |
| ~100 min | **`03_evaluation.ipynb`** | Domain-aware evaluation: vendor-metric critique → open + axial coding → synthetic data + golden dataset → programmatic + DSPy-optimized LLM-judge → CI pipeline → online monitoring + drift. |
| ~30 min | breaks + Q&A | |

See **`docs/agent-flow.html`** in your browser for a visual overview of the agent — open it locally:

```bash
open docs/agent-flow.html         # macOS
xdg-open docs/agent-flow.html     # Linux
```

---

## 8. Troubleshooting

### `uv sync` fails
- **`onnxruntime` wheel error on macOS** — make sure you're on macOS 13 or newer. Update Xcode command-line tools: `xcode-select --install`.
- **Network/proxy errors** — try a fresh shell or set `HTTPS_PROXY` if you're behind a corporate proxy.

### `baml-cli generate` fails
- Make sure `uv sync --all-extras` finished cleanly first; `baml-py` brings the CLI.
- On Apple Silicon, ensure Python 3.11+ (`uv` handles this automatically). Older Pythons fail to install `baml-py`.

### Phoenix port already in use
- The smoke notebook prints a `⚠️` and the workshop falls back to a JSON trace logger. You can still complete every section.
- To clear a stuck port: `lsof -ti :6006 | xargs kill -9` (then rerun the cell).

### OpenAI 429 (rate limit)
- The workshop uses `gpt-4o-mini` (very low quota).
- Every LLM call is cached to disk under `.cache/`, so re-runs are free. Wait 60s and re-run only the failing cell.

### `baml_client` import error
- Run `uv run baml-cli generate --from baml_src` (step 4 above). The generated client is gitignored, so it's missing on a fresh clone.

### PDF extraction returns empty text
- If you see `nanny_resume_*.pdf` extract to empty strings, restore with `git checkout data/pdfs/`.

### A notebook cell crashes with `NameError`
- The notebooks are designed to run **top-to-bottom**. If you jump in mid-section, restart the kernel and run all cells from the start: **Kernel → Restart & Run All**.

### "I want to start fresh"
- Delete `nanny_db/`, `.cache/`, `data/memory.json`, and `baml_client/`. They'll be rebuilt on the next run. Your `.env` and `data/seed_db.json` stay.

---

## 9. What to bring on the day

- A laptop with the setup above completed and the smoke test passing.
- Your OpenAI key in `.env`.
- A modern browser (Chrome, Firefox, Safari).
- A second screen if you have one — useful for keeping the Phoenix UI open beside the notebook.
- Questions about your own LLM systems — we'll discuss them between sections.

See you at the workshop! 👋
