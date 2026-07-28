# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beautifulsoup4>=4.12.3",
#     "bm25s>=0.1.10",
#     "cohere>=5.13.6",
#     "marimo>=0.23.3",
#     "nltk>=3.8.1",
#     "numpy<2.0.0",
#     "pymdown-extensions>=10.8.1",
#     "pystemmer>=2.2.0.1",
#     "python-dotenv>=1.0.1",
#     "python-frontmatter>=1.1.0",
#     "requests>=2.31.0",
#     "rich>=13.0.0",
#     "scikit-learn>=1.5.0",
#     "scipy>=1.14.0",
#     "wandb>=0.17.7",
#     "weave>=0.51.27",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5: Retrieval and Re-ranking

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter05.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter05.py)

    <!--- @wandbcode{rag-course-05} -->

    In this chapter, we will focus on improving the retrieval quality and employ reranking to select the best context for our LLM.

    In the last chapter, we added a Query Enhancement module which was able to derive sub-queries from the user query. We retrieved contexts for these sub-queries. We did a basic deduplication of the total context to reduce the same context. However on a semantic level there might be few more contexts that we can drop thus reducing the input token cost.

    When the same context is retrieved a few times, we can use this knowledge to fuse the confidence scores of such contexts and ask our LLM to focus more on them. One can employ many strategies to improve the quality of contexts we provide our LLM.

    Let's see a few here.

    This is a [marimo](https://marimo.io) notebook. Cohere trial keys: chat **20 req/min**, **Rerank 10 req/min**, **1,000 calls/month**. Indexing embeddings over the full chunk set is expensive — every Cohere-spending step is gated behind a button.
    """)
    return


@app.cell
def _(mo):
    import os
    import pathlib
    import sys
    import urllib.request

    NOTEBOOKS_DIR = pathlib.Path(mo.notebook_dir()).resolve().parent / "notebooks"
    DATA_DIR = pathlib.Path(mo.notebook_dir()).resolve().parent / "data"

    _REPO_RAW = "https://raw.githubusercontent.com/wandb/edu/feat/rag-advanced-marimo-pilot/rag-advanced/notebooks"
    _SCRIPT_FILES = [
        "__init__.py",
        "chunking.py",
        "embedding.py",
        "preprocess.py",
        "query_enhancer.py",
        "rag_pipeline.py",
        "reranker.py",
        "response_generator.py",
        "response_metrics.py",
        "retrieval_metrics.py",
        "retriever.py",
        "utils.py",
    ]
    if not (NOTEBOOKS_DIR / "scripts" / "__init__.py").exists():
        (NOTEBOOKS_DIR / "scripts").mkdir(parents=True, exist_ok=True)
        for _name in _SCRIPT_FILES:
            urllib.request.urlretrieve(
                f"{_REPO_RAW}/scripts/{_name}", NOTEBOOKS_DIR / "scripts" / _name
            )

    if str(NOTEBOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS_DIR))
    return DATA_DIR, NOTEBOOKS_DIR, os, pathlib


@app.cell(hide_code=True)
def _(mo):
    def show_source(symbol):
        import inspect

        source = inspect.getsource(symbol)
        return mo.accordion(
            {f"Source of `{symbol.__name__}`": mo.md(f"```python\n{source}\n```")},
            lazy=True,
        )

    def preview_table(rows, n=3):
        flat = []
        for _row in list(rows)[:n]:
            _row = dict(_row)
            _entry = {k: v for k, v in _row.items() if k != "metadata"}
            _entry.update(_row.get("metadata", {}))
            flat.append(_entry)
        return mo.ui.table(flat, selection=None)

    return preview_table, show_source


@app.cell
def _():
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    return


@app.cell
def _(mo, os):
    def _saved_wandb_api_key():
        try:
            import netrc

            auth = netrc.netrc().authenticators("api.wandb.ai")
            return auth[2] if auth else None
        except Exception:
            return None

    cohere_key_input = mo.ui.text(
        label="Cohere API key ([dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys))",
        kind="password",
    )
    wandb_key_input = mo.ui.text(
        label="W&B API key ([wandb.ai/authorize](https://wandb.ai/authorize))",
        kind="password",
    )
    wandb_key_found = bool(os.environ.get("WANDB_API_KEY") or _saved_wandb_api_key())

    _needed = []
    if not os.environ.get("COHERE_API_KEY"):
        _needed.append(cohere_key_input)
    if not wandb_key_found:
        _needed.append(wandb_key_input)
    (
        mo.vstack(_needed)
        if _needed
        else mo.callout("API credentials found — you are ready to go.", kind="success")
    )
    return cohere_key_input, wandb_key_found, wandb_key_input


@app.cell
def _(cohere_key_input, mo, os, wandb_key_found, wandb_key_input):
    cohere_api_key = os.environ.get("COHERE_API_KEY") or cohere_key_input.value
    mo.stop(
        not (cohere_api_key and (wandb_key_found or wandb_key_input.value)),
        mo.callout("Enter your API key(s) above to continue.", kind="warn"),
    )
    os.environ["COHERE_API_KEY"] = cohere_api_key
    os.environ["CO_API_KEY"] = cohere_api_key
    os.environ.setdefault("WEAVE_PARALLELISM", "1")
    if wandb_key_input.value:
        os.environ["WANDB_API_KEY"] = wandb_key_input.value
    api_keys_ready = True
    return (api_keys_ready,)


@app.cell
def _(api_keys_ready, mo):
    mo.stop(not api_keys_ready)
    import weave

    WANDB_PROJECT = "rag-course"
    weave_client = weave.init(WANDB_PROJECT)
    return WANDB_PROJECT, weave, weave_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will download the chunked data from chapter 3.
    """)
    return


@app.cell
def _(mo):
    chunked_data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (chunked_data:latest)": "own",
        },
        value="Course dataset (public Weave ref shared by the course)",
        label="Fetch chunked_data from:",
    )
    chunked_data_source
    return (chunked_data_source,)


@app.cell
def _(chunked_data_source, mo, preview_table, weave, weave_client):
    mo.stop(weave_client is None)
    _chunked_ref = (
        "weave:///wandb-smle/rag-course/object/chunked_data:CRbEh79BgxWPqbD1rOozLNw12pky0LhQYF2PIXgc2pQ"
        if chunked_data_source.value == "course"
        else "chunked_data:latest"
    )
    with mo.status.spinner(title="Fetching `chunked_data` from Weave..."):
        _chunked_ds = weave.ref(_chunked_ref).get()
    chunked_data = list(map(dict, _chunked_ds.rows[:]))
    mo.vstack(
        [
            mo.md(f"**Chunks:** {len(chunked_data)}  \nRef: `{_chunked_ref}`"),
            preview_table(chunked_data, n=2),
        ]
    )
    return (chunked_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding based retriever

    In the past chapters we were using TF-IDF and BM25 based retrievers. The vector representation from these methods are not "dense" i.e, they were not trained on billions or trillions of tokens. LLMs today embed the tokens into a dense representation. These embeddings are learned during the training process and are used to represent the tokens in the high-dimensional vector space. They have a denser knowledge of different linguistic patterns.

    One straightforward way of improving our retriever is to use a `DenseRetriever` which uses the embedding model (Cohere embedding model here) to embed the chunks and uses the same embedding model to embed the query.

    ## Reranking the contexts

    With more contexts it is important to pick the one that adds more knowledge about the given query. For this, a re-ranking model is used that calculates a matching score for a given query and document pair. This score can then be utilized to rearrange vector search results, ensuring that the most relevant results are prioritized at the top of the list. Cohere comes with its own re-ranking model (`rerank-english-v3.0`) and is quite popular.

    Trial key note: **Rerank is limited to 10 req/min** — evaluations that call the reranker will be slower than chat-only steps.
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.reranker import CohereReranker
    from scripts.retriever import DenseRetriever, DenseRetrieverWithReranker

    mo.accordion(
        {
            "`DenseRetriever`": show_source(DenseRetriever),
            "`CohereReranker`": show_source(CohereReranker),
            "`DenseRetrieverWithReranker`": show_source(DenseRetrieverWithReranker),
        },
        lazy=True,
    )
    return CohereReranker, DenseRetriever, DenseRetrieverWithReranker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's initialize the `DenseRetriever` and index the data.

    Indexing embeds **every chunk** via Cohere (`embed-english-v3.0`) — this is a large API spend. Click only when you intend to run it.
    """)
    return


@app.cell
def _(mo):
    dense_index_button = mo.ui.run_button(
        label=f"Index DenseRetriever (embeds all chunks via Cohere)"
    )
    dense_index_button
    return (dense_index_button,)


@app.cell
def _(DenseRetriever, chunked_data, dense_index_button, mo):
    mo.stop(
        not dense_index_button.value,
        mo.md("*Click the button above to embed and index chunks for dense retrieval.*"),
    )
    dense_retriever = DenseRetriever()
    with mo.status.spinner(
        title=f"Embedding + indexing {len(chunked_data)} chunks (Cohere)..."
    ):
        dense_retriever.index_data(chunked_data)
    mo.callout(
        mo.md(f"`DenseRetriever` indexed **{len(chunked_data)}** chunks."),
        kind="success",
    )
    return (dense_retriever,)


@app.cell
def _(mo):
    eval_data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (eval_data:latest)": "own",
        },
        value="Course dataset (public Weave ref shared by the course)",
        label="Fetch eval dataset from:",
    )
    eval_data_source
    return (eval_data_source,)


@app.cell
def _(eval_data_source, mo, preview_table, weave, weave_client):
    mo.stop(weave_client is None)
    _eval_ref = (
        "weave:///wandb-smle/rag-course/object/Dataset:0Vva0o2YhyGG6sh4yGf1b5woW9teuXP8zEquFTTcgfQ"
        if eval_data_source.value == "course"
        else "eval_data:latest"
    )
    with mo.status.spinner(title="Fetching eval dataset..."):
        eval_dataset = weave.ref(_eval_ref).get()
    mo.vstack(
        [
            mo.md(
                f"**Evaluation samples:** {len(eval_dataset.rows)}  \nRef: `{_eval_ref}`"
            ),
            preview_table(eval_dataset.rows, n=1),
        ]
    )
    return (eval_dataset,)


@app.cell
def _():
    from scripts.retrieval_metrics import IR_METRICS

    return (IR_METRICS,)


@app.cell
def _(mo):
    dense_eval_button = mo.ui.run_button(
        label="Evaluate DenseRetriever (IR metrics; embeds queries via Cohere)"
    )
    dense_eval_button
    return (dense_eval_button,)


@app.cell
async def _(
    IR_METRICS,
    dense_eval_button,
    dense_retriever,
    eval_dataset,
    mo,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not dense_eval_button.value,
        mo.md("*Click the button above to evaluate the dense retriever.*"),
    )
    dense_retrieval_eval = weave.Evaluation(
        name="Dense Retrieval Evaluation",
        dataset=eval_dataset,
        scorers=IR_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"], "k": 10},
    )
    with mo.status.spinner(title="Evaluating DenseRetriever..."):
        dense_retrieval_scores = await dense_retrieval_eval.evaluate(dense_retriever)
    mo.md(f"```python\n{dense_retrieval_scores!r}\n```")
    return dense_retrieval_eval, dense_retrieval_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's initialize the `DenseRetrieverWithReranker` and index the data.

    Indexing again embeds all chunks (same cost profile as above). Evaluation also hits the **Rerank** endpoint (10 req/min on trial keys).
    """)
    return


@app.cell
def _(mo):
    rerank_index_button = mo.ui.run_button(
        label="Index DenseRetrieverWithReranker (embeds all chunks via Cohere)"
    )
    rerank_index_button
    return (rerank_index_button,)


@app.cell
def _(DenseRetrieverWithReranker, chunked_data, mo, rerank_index_button):
    mo.stop(
        not rerank_index_button.value,
        mo.md("*Click the button above to index the dense+rerank retriever.*"),
    )
    dense_retriever_rerank = DenseRetrieverWithReranker()
    with mo.status.spinner(
        title=f"Embedding + indexing {len(chunked_data)} chunks for dense+rerank..."
    ):
        dense_retriever_rerank.index_data(chunked_data)
    mo.callout(
        mo.md(
            f"`DenseRetrieverWithReranker` indexed **{len(chunked_data)}** chunks "
            "(rerank model: `rerank-english-v3.0`)."
        ),
        kind="success",
    )
    return (dense_retriever_rerank,)


@app.cell
def _(mo):
    dense_rerank_eval_button = mo.ui.run_button(
        label="Evaluate DenseRetrieverWithReranker (Cohere embed + rerank @ 10/min)"
    )
    dense_rerank_eval_button
    return (dense_rerank_eval_button,)


@app.cell
async def _(
    IR_METRICS,
    dense_rerank_eval_button,
    dense_retriever_rerank,
    eval_dataset,
    mo,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not dense_rerank_eval_button.value,
        mo.md("*Click the button above to evaluate dense retrieval + rerank.*"),
    )
    dense_rerank_eval = weave.Evaluation(
        name="Dense Retrieval Rerank Evaluation",
        dataset=eval_dataset,
        scorers=IR_METRICS,
        preprocess_model_input=lambda x: {
            "query": x["question"],
            "top_k": 20,
            "top_n": 10,
        },
    )
    with mo.status.spinner(
        title="Evaluating DenseRetrieverWithReranker (rerank @ ~10 req/min)..."
    ):
        dense_rerank_scores = await dense_rerank_eval.evaluate(dense_retriever_rerank)
    mo.md(f"```python\n{dense_rerank_scores!r}\n```")
    return dense_rerank_eval, dense_rerank_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hybrid Retriever

    Even though BM25 is an old model used for retrieval tasks, it is still the state-of-the-art on various benchmarks. In machine learning, we ensemble a few weak classifiers to build a stronger classifier; we can adopt the same idea to our retriever pipeline.

    Below we show the concept of hybrid retriever which uses two or more retrievers and retrieves chunks from all of them followed by re-ranking.

    ### Fusion reranking

    Since the individual retrievers are going to retrieve the same chunks most of the time, simple re-ranking will not be beneficial since the same chunk from each retriever will have a similar score, thus the top_k after re-ranking will have more or less the same context.

    Instead, we iterate through all the chunks and fuse the score of similar chunks. Finally we sort on the basis of this fused score and return top_k chunks.
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.retriever import HybridRetrieverReranker

    show_source(HybridRetrieverReranker)
    return (HybridRetrieverReranker,)


@app.cell
def _(mo):
    hybrid_index_button = mo.ui.run_button(
        label="Index HybridRetrieverReranker (embeds all chunks via Cohere)"
    )
    hybrid_index_button
    return (hybrid_index_button,)


@app.cell
def _(HybridRetrieverReranker, chunked_data, hybrid_index_button, mo):
    mo.stop(
        not hybrid_index_button.value,
        mo.md("*Click the button above to index the hybrid retriever.*"),
    )
    hybrid_retriever = HybridRetrieverReranker()
    with mo.status.spinner(
        title=f"Indexing hybrid retriever over {len(chunked_data)} chunks..."
    ):
        hybrid_retriever.index_data(chunked_data)
    mo.callout(
        mo.md(f"`HybridRetrieverReranker` indexed **{len(chunked_data)}** chunks."),
        kind="success",
    )
    return (hybrid_retriever,)


@app.cell
def _(mo):
    hybrid_eval_button = mo.ui.run_button(
        label="Evaluate HybridRetrieverReranker (Cohere embed + rerank @ 10/min)"
    )
    hybrid_eval_button
    return (hybrid_eval_button,)


@app.cell
async def _(
    IR_METRICS,
    eval_dataset,
    hybrid_eval_button,
    hybrid_retriever,
    mo,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not hybrid_eval_button.value,
        mo.md("*Click the button above to evaluate the hybrid retriever.*"),
    )
    hybrid_retrieval_eval = weave.Evaluation(
        name="Hybrid Retrieval Rerank Evaluation",
        dataset=eval_dataset,
        scorers=IR_METRICS,
        preprocess_model_input=lambda x: {
            "query": x["question"],
            "top_k": 20,
            "top_n": 10,
        },
    )
    with mo.status.spinner(
        title="Evaluating HybridRetrieverReranker (rerank @ ~10 req/min)..."
    ):
        hybrid_retrieval_scores = await hybrid_retrieval_eval.evaluate(hybrid_retriever)
    mo.md(f"```python\n{hybrid_retrieval_scores!r}\n```")
    return hybrid_retrieval_eval, hybrid_retrieval_scores


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.callout(
        mo.md(
            f"""
    ## Key Takeaways

    - Dense embeddings capture richer linguistic patterns than TF-IDF/BM25 alone.
    - Cohere reranking (`rerank-english-v3.0`) reorders candidates by query–document relevance.
    - Hybrid retrieval + fusion reranking combines lexical and dense signals.
    - Trial-key limits matter here: embeddings for indexing and **10 rerank req/min** dominate wall-clock time.

    Inspect runs in [**your Weave evaluations**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/evaluations).
    """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
