# Marimo Conversion Design — Template for Edu Revamp

**Status:** Pilot ready for review (rag-advanced Chapter00 + Chapter01)  
**Date:** 2026-07-28  
**Scope:** Convert Jupyter/Colab course notebooks to marimo so they run identically locally (`uvx marimo edit --sandbox`) and on [molab](https://molab.marimo.io). Course content stays 1:1; only execution mechanics and presentation change.  
**Reference implementation:** `rag-advanced/marimo/Chapter00.py`, `rag-advanced/marimo/Chapter01.py`

This document is the reusable template proposal for converting other courses under `wandb/edu`. Approve (or amend) these conventions before batch-converting the remaining rag-advanced chapters.

---

## Goals and non-goals

**Goals**

- One `.py` marimo notebook per chapter, with PEP 723 inline deps — same file locally and on molab.
- No Colab `!pip` / `%cd` / `getpass` / `wandb.login()` ceremony.
- Shared course code (`scripts/`, `prompts/`) remains importable both in-repo and when molab opens only the single notebook file.
- Better-than-print presentation: tables, accordions, callouts, gated API spend, concrete Weave UI links.
- Cohere free-trial limits and dated model names are handled so notebooks keep working after the Sept 2025 deprecations.

**Non-goals**

- No rewrites of course prose or pedagogy.
- No changes to the original `.ipynb` files.
- No logic changes in `notebooks/scripts/` beyond marimo compatibility (plain iterators instead of `tqdm.notebook`, dual Cohere env key names).

---

## File layout

```
rag-advanced/
├── notebooks/                 # original .ipynb + scripts/ + prompts/ (unchanged content)
├── marimo/                    # marimo editions
│   ├── Chapter00.py
│   ├── Chapter01.py
│   └── …                      # Phase 2: Chapter02–06, rag_tooluse_v2
└── README.md                  # open-in-molab badges + run instructions
```

**Scaffold command**

```bash
uvx marimo@latest convert notebooks/ChapterXX.ipynb -o marimo/ChapterXX.py
```

Then refactor for reactivity, setup, and presentation (do not ship the raw convert output).

---

## 1. File format and dependencies

Every notebook starts with a PEP 723 header. Maintain it via `--sandbox` so local and molab installs stay in sync:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cohere>=5.13.6",
#     "marimo>=0.23.3",
#     "weave>=0.51.27",
#     # chapter-specific packages…
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")
```

- Declare only real third-party packages. Never list `scripts` as a pip dependency (molab treats that as a missing package).
- Chapter-specific extras go in that chapter’s header only (e.g. LangChain/Tavily for `rag_tooluse_v2`).
- Prefer `app = marimo.App(width="medium")` unless a chapter needs full width for wide tables.

**Local run**

```bash
cd rag-advanced/marimo
uvx marimo edit --sandbox Chapter01.py
# or headless:
uvx marimo run --sandbox Chapter01.py
```

---

## 2. Setup cell pattern (identical across chapters)

### 2.1 API keys

Order of precedence:

1. `os.environ` (local `.env` / shell / molab Secrets panel)
2. Fallback `mo.ui.text(kind="password")` widgets
3. `mo.stop(...)` until required keys exist — blocks all downstream cells

Always set both Cohere env names (scripts disagree on which they read):

```python
os.environ["COHERE_API_KEY"] = cohere_api_key
os.environ["CO_API_KEY"] = cohere_api_key
os.environ.setdefault("WEAVE_PARALLELISM", "1")  # trial keys: 20 chat req/min
```

W&B: accept `WANDB_API_KEY` from the environment, or detect an existing `~/.netrc` login for `api.wandb.ai`, else show a password field. Do not call `wandb.login()` or `getpass`.

Show a success `mo.callout` when credentials are already present so returning learners are not prompted again.

### 2.2 Weave init

Call `weave.init("rag-course")` (or the chapter’s `WANDB_PROJECT`) in **exactly one cell**, after keys are ready. Gate with `mo.stop` on the keys-ready flag.

### 2.3 Shared `scripts/` + `prompts/` bootstrap

In-repo: resolve `notebooks/` via `mo.notebook_dir()` and insert it on `sys.path`:

```python
NOTEBOOKS_DIR = pathlib.Path(mo.notebook_dir()).resolve().parent / "notebooks"
DATA_DIR = pathlib.Path(mo.notebook_dir()).resolve().parent / "data"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))
```

On molab (verified): opening from GitHub imports **only** the single notebook file. When `scripts/__init__.py` is absent, the setup cell downloads the needed files from GitHub raw:

```python
_REPO_RAW = (
    "https://raw.githubusercontent.com/wandb/edu/"
    "feat/rag-advanced-marimo-pilot/rag-advanced/notebooks"
)
# After the PR merges, switch the branch segment to `main`.
```

Bootstrap only the files that chapter needs (scripts modules + prompt text files). Do not `pip install` the course package.

---

## 3. Reactivity refactors (mechanical)

Marimo forbids redefining the same name across cells. When converting:

| Jupyter pattern | Marimo pattern |
|---|---|
| Reuse `eval`, `scores`, `results` across sections | Distinct names: `eval_tfidf`, `eval_bm25`, `scores_v1`, … |
| Mutate a `messages` list in place across cells | Explicit per-step variables (`messages_turn_1`, `messages_turn_2`, …) |
| `nest_asyncio` + `asyncio.run(...)` | Top-level `await` in the cell |
| Cells that only work under IPython | Drop magics (`%%wandb`, `%cd`, `!pip`) |

Heaviest rename work: Chapters 02, 05, 06. `rag_tooluse_v2` needs the conversation-state restructure.

---

## 4. Presentation conventions

| Instead of… | Use… |
|---|---|
| `print` / `pprint` of results | `mo.md(f"…")` narration |
| Key takeaways / warnings | `mo.callout(..., kind="success"\|"warn"\|"danger")` |
| Lists of dicts / eval scores | `mo.ui.table(..., selection=None)` |
| Rich `display_source` | `mo.accordion({f"Source of `{name}`": mo.md(f"```python\n{src}\n```")})` (prefer `lazy=True` for large sources) |
| `IPython.display.Markdown(response)` | `mo.md(response)` |
| Prose/image-only pipeline diagrams | `mo.mermaid("""…""")` where it clarifies flow |
| `tqdm.notebook` progress | `mo.status.progress_bar` / `mo.status.spinner` for long steps |
| Console “🍩” Weave URL | Concrete markdown link from `weave_client`: `https://wandb.ai/{entity}/{project}/weave/traces` |
| Commented “uncomment to use your project” | `mo.ui.radio` data-source switch (`course` public Weave ref vs `name:latest` in the learner’s project) |

### 4.1 Cost gating and caching

Every Cohere-spending cell (chat, generation, evals, rerank) must be gated:

```python
run_btn = mo.ui.run_button(label="Generate answer (calls Cohere)")
# …
mo.stop(not run_btn.value, mo.md("*Click the button above to …*"))
```

Wrap expensive, idempotent steps (embeddings, eval result objects) in `mo.persistent_cache` so learners pay once per machine/session where feasible.

For trial-key eval chapters (02–06), add a short note near each eval button: expected wall-clock under 20 req/min (and 10 req/min for Cohere Rerank in Ch05).

### 4.2 Data-source radio (Ch01+)

```python
data_source = mo.ui.radio(
    options={
        "Course dataset (public Weave ref shared by the course)": "course",
        "My project (the datasets you published above)": "own",
    },
    value="Course dataset (public Weave ref shared by the course)",
    label="Fetch Weave Datasets from:",
)
# then:
ref = (
    "weave:///wandb-smle/rag-course/object/…"
    if data_source.value == "course"
    else "raw_data:latest"
)
```

Reactivity re-runs every dependent cell when the learner toggles. Apply to **every** hardcoded Weave-ref fetch in Chapters 02–06.

### 4.3 Open-in-molab header

Each notebook’s intro markdown includes:

```markdown
[Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/ChapterXX.py)
· [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/ChapterXX.py)
```

README badges use the official shield:

```markdown
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/ChapterXX.py)
```

---

## 5. Cohere: trial limits and model names

**Trial key (docs.cohere.com, Jul 2026)**

- Signup at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) auto-issues a free trial key (old Google Form is dead).
- Limits: chat **20 req/min**, Rerank **10 req/min**, **1,000 API calls/month**.
- Notebook copy must mention these limits; setup always sets `WEAVE_PARALLELISM=1`.

**Model mapping (aliases deprecated 2025-09-15)**

| Deprecated alias | Replacement |
|---|---|
| `command-r` | `command-r-08-2024` |
| `command-r-plus` | `command-r-plus-08-2024` |
| `command` / `command-light` | do not use; pick a dated Command snapshot |

Applied in the pilot notebooks and shared scripts (`query_enhancer.py`, `response_metrics.py`, `utils.py` defaults; `retrieval_metrics.py` already dated). Phase 2: same mapping everywhere, including Ch06’s upgrade narrative and Ch05 rerank model IDs (`rerank-english-v3.0` / `rerank-multilingual-v3.0` — re-verify at conversion time).

---

## 6. Shared helpers (per-notebook, light)

Chapters that import `scripts.*` should define small display helpers once near the top (see Chapter01):

- `show_source(symbol)` — accordion with syntax-highlighted source
- `preview_table(rows, n=3)` — flatten `content` + `metadata` into `mo.ui.table`

Keep helpers inside the notebook (not a new shared package) so molab single-file open stays simple.

---

## 7. Conversion checklist (per chapter)

1. `uvx marimo@latest convert …` → `marimo/ChapterXX.py`
2. Add/refresh PEP 723 deps; drop Colab/IPython-only cells
3. Install the setup pattern: keys → `WEAVE_PARALLELISM` → `weave.init` → scripts bootstrap
4. Split redefined names; replace async wrappers with top-level `await`
5. Map Cohere model aliases to dated snapshots
6. Replace prints/pprints with `mo.md` / `mo.ui.table` / `mo.callout`
7. Gate every paid API cell with `mo.ui.run_button` + `mo.stop`; cache where useful
8. Replace “uncomment own project” with `mo.ui.radio`
9. Render Weave UI links from `weave_client.entity` / `.project`
10. Add molab + GitHub links in the intro cell
11. Verify: `uvx marimo edit --sandbox ChapterXX.py` top-to-bottom; smoke-test on molab (secrets, scripts bootstrap, Weave traces)

---

## 8. Pilot status vs Phase 2

| Convention | Ch00 | Ch01 | Phase 2 notes |
|---|---|---|---|
| PEP 723 + sandbox | ✅ | ✅ | Required |
| Keys + `mo.stop` + dual Cohere env | ✅ | ✅ | Required |
| `WEAVE_PARALLELISM=1` | ✅ | ✅ | Required |
| Scripts/prompts GitHub bootstrap | n/a | ✅ | Pin branch → `main` on merge |
| `mo.ui.run_button` cost gating | ✅ | ✅ | All eval/generation cells |
| `mo.ui.table` / accordion / callout | light | ✅ | Expand in eval chapters |
| `mo.ui.radio` data-source | n/a | ✅ | Every Weave-ref fetch |
| Weave UI markdown links | ✅ | ✅ | Required |
| Dated Cohere models | ✅ | ✅ | Ch02–06 + tooluse |
| `mo.mermaid` pipeline diagrams | — | deferred | Add where prose describes flow |
| `mo.persistent_cache` around spend | — | deferred | Priority for Ch02–06 evals |
| Top-level `await` (drop nest_asyncio) | n/a | n/a | Ch04+ |
| Distinct eval variable names | n/a | n/a | Ch02/05/06 heavy |

**Open risk:** Weave tracing on molab had no public prior example; Chapter00 is the canary. If tracing fails on molab, local runs remain the source of truth and notebooks still link out to the Weave UI.

---

## 9. Out of scope (reminders)

- No course-content or `.ipynb` edits
- No `scripts/` logic changes beyond tqdm-iterator compatibility and Cohere key-name tolerance
- No force-install of course code as a pip package on molab

---

## 10. Sign-off

Reviewers should approve this template (or list amendments) before Phase 2 batch conversion of Chapters 02–06 and `rag_tooluse_v2`. See also the short learner-facing section in [`rag-advanced/README.md`](../../../rag-advanced/README.md).
