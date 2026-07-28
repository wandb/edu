# RAG++

This repository contains the code for the [RAG++ : From POC to Production course](https://www.wandb.courses/courses/rag-in-production).

**LLM-friendly Resources**

To help learn with ChatGPT etc, full text concatentation of the entire course transcript and course notebooks [can be found here](https://github.com/wandb/edu/tree/main/rag-advanced/resources#llm-friendly-resources)

# Course Overview

Practical RAG techniques for engineers: learn production-ready solutions from industry experts to optimize performance, cut costs, and enhance the accuracy and relevance of your applications.

## Marimo notebooks (pilot)

Reactive [marimo](https://marimo.io) editions of the course notebooks live in [`marimo/`](marimo/). They use PEP 723 inline dependencies, so the same file runs locally and on [molab](https://molab.marimo.io) — no Colab `!pip` cells.

| Chapter | molab | Source |
| --- | --- | --- |
| Chapter 0 — Setup | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter00.py) | [`marimo/Chapter00.py`](marimo/Chapter00.py) |
| Chapter 1 — Simple RAG | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter01.py) | [`marimo/Chapter01.py`](marimo/Chapter01.py) |
| Chapter 2 — Evaluation | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter02.py) | [`marimo/Chapter02.py`](marimo/Chapter02.py) |
| Chapter 3 — Ingestion & preprocessing | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter03.py) | [`marimo/Chapter03.py`](marimo/Chapter03.py) |
| Chapter 4 — Query enhancement | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter04.py) | [`marimo/Chapter04.py`](marimo/Chapter04.py) |
| Chapter 5 — Retrieval & reranking | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter05.py) | [`marimo/Chapter05.py`](marimo/Chapter05.py) |
| Chapter 6 — Response synthesis | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter06.py) | [`marimo/Chapter06.py`](marimo/Chapter06.py) |

`rag_tooluse_v2` still follows the same template (not yet converted). Design conventions: [`docs/superpowers/specs/2026-07-28-marimo-conversion-design.md`](../docs/superpowers/specs/2026-07-28-marimo-conversion-design.md).

**Run locally** (requires [uv](https://docs.astral.sh/uv/)):

```bash
cd rag-advanced/marimo
uvx marimo edit --sandbox Chapter00.py   # setup + Weave + Cohere smoke test
uvx marimo edit --sandbox Chapter01.py   # full Chapter 1 pipeline
uvx marimo edit --sandbox Chapter02.py   # retrieval + response evaluations
uvx marimo edit --sandbox Chapter03.py   # preprocessing, BM25 vs TF-IDF
uvx marimo edit --sandbox Chapter04.py   # query enhancement vs baseline
uvx marimo edit --sandbox Chapter05.py   # dense retrieval + Cohere rerank
uvx marimo edit --sandbox Chapter06.py   # prompt iteration + model upgrade
```

Set `COHERE_API_KEY` and `WANDB_API_KEY` in the environment (or a repo-root `.env`), or enter them in the notebook password fields. On molab, use the Secrets panel. Cohere trial keys are rate-limited (20 chat req/min, 1,000 calls/month); the notebooks keep Weave evaluation concurrency at 1 to stay under the limit.

The original Jupyter notebooks under `notebooks/` are unchanged.

## Setup and Installation

First create an environment with python 3.10:
```bash
conda create -n rag-advanced python=3.10
conda activate rag-advanced
```

Then install the requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

The project is structured as follows:

```
├── data      # contains the data used for the course
├── notebooks # original Jupyter notebooks, scripts/, and prompts/
├── marimo    # marimo editions (pilot: Chapter00–01)
├── src       # contains the source code for the course
```
