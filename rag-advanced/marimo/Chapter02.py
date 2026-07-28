# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beautifulsoup4>=4.12.3",
#     "bm25s>=0.1.10",
#     "cohere>=5.13.6",
#     "levenshtein>=0.25.1",
#     "marimo>=0.23.3",
#     "nltk>=3.8.1",
#     "numpy<2.0.0",
#     "pydantic>=2.0.0",
#     "pymdown-extensions>=10.8.1",
#     "pystemmer>=2.2.0.1",
#     "python-dotenv>=1.0.1",
#     "python-frontmatter>=1.1.0",
#     "requests>=2.31.0",
#     "rich>=13.0.0",
#     "rouge>=1.0.1",
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
    # Chapter 2: Comprehensive Evaluation Strategies

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter02.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter02.py)

    <!--- @wandbcode{rag-course-02} -->

    In this chapter, we will evaluate the two main components of a RAG pipeline — retriever and response generator.

    Evaluating the retriever can be considered component evaluation. Depending on your RAG pipeline, there can be a few components and for ensuring robustness of your system, it is recommended to come up with evaluation for each component.

    This is a [marimo](https://marimo.io) notebook: its dependencies are declared inline
    at the top of the file and are installed automatically when you open it with
    `uvx marimo edit --sandbox Chapter02.py` or on [molab](https://molab.marimo.io).

    You will need a Cohere API key and a Weights & Biases account. Cohere trial keys are
    rate-limited to **20 chat requests/minute** and **1,000 API calls/month** — the
    evaluation cells below make many LLM calls, so each run is gated behind a button
    and Weave concurrency is kept at 1. Expect several minutes per evaluation on a trial key.
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
    _PROMPT_FILES = ["initial_system.txt"]
    if not (NOTEBOOKS_DIR / "scripts" / "__init__.py").exists():
        (NOTEBOOKS_DIR / "scripts").mkdir(parents=True, exist_ok=True)
        (NOTEBOOKS_DIR / "prompts").mkdir(parents=True, exist_ok=True)
        for _name in _SCRIPT_FILES:
            urllib.request.urlretrieve(
                f"{_REPO_RAW}/scripts/{_name}", NOTEBOOKS_DIR / "scripts" / _name
            )
        for _name in _PROMPT_FILES:
            urllib.request.urlretrieve(
                f"{_REPO_RAW}/prompts/{_name}", NOTEBOOKS_DIR / "prompts" / _name
            )

    if str(NOTEBOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS_DIR))
    return DATA_DIR, NOTEBOOKS_DIR, os, pathlib


@app.cell(hide_code=True)
def _(mo):
    def show_source(symbol):
        """Render a symbol's source code in a collapsible accordion."""
        import inspect

        source = inspect.getsource(symbol)
        return mo.accordion(
            {f"Source of `{symbol.__name__}`": mo.md(f"```python\n{source}\n```")},
            lazy=True,
        )

    def preview_table(rows, n=3):
        """Show the first `n` document dicts as an interactive table."""
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will start by setting up the API keys. Get your Cohere API key from
    [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and your
    W&B key from [wandb.ai/authorize](https://wandb.ai/authorize).
    """)
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
    # Trial keys: 20 chat req/min. Keep Weave evaluations sequential.
    os.environ.setdefault("WEAVE_PARALLELISM", "1")
    if wandb_key_input.value:
        os.environ["WANDB_API_KEY"] = wandb_key_input.value
    api_keys_ready = True
    return (api_keys_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this chapter we will also use W&B Weave for our evaluation purposes. The `weave.Evaluation` class is a light weight class that can be used to evaluate the performance of a `weave.Model` on a `weave.Dataset`. We will go into more details.

    We first initialize a weave client which can track both the traces and the evaluation scores.
    """)
    return


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
    ## Collecting data for evaluation

    We are using a subset of the evaluation dataset we had created for wandbot.

    Learn more about how we created the evaluation dataset here:

    - [How to Evaluate an LLM, Part 1: Building an Evaluation Dataset for our LLM System](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-Evaluate-an-LLM-Part-1-Building-an-Evaluation-Dataset-for-our-LLM-System--Vmlldzo1NTAwNTcy)
    - [How to Evaluate an LLM, Part 2: Manual Evaluation of Wandbot, our LLM-Powered Docs Assistant](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-Evaluate-an-LLM-Part-2-Manual-Evaluation-of-Wandbot-our-LLM-Powered-Docs-Assistant--Vmlldzo1NzU4NTM3)

    The main take away from these reports are:

    - we first deployed wandbot for internal usage based on rigorous eyeballing based evalution.
    - the user query distribution was throughly analyzed and clustered. we sampled a good representative queries from these clusters and created a gold standard set of queries.
    - we then used in-house MLEs to perform manual evaluation using Argilla. Creating such evaluation platforms are easy.
    - To summarize, speed is the key here. Use whatever means you have to create a meaningful eval set.

    The evaluation samples are logged as [`weave.Dataset`](https://wandb.github.io/weave/guides/core-types/datasets/). `weave.Dataset` enables you to collect examples for evaluation and automatically track versions for accurate comparisons.

    Below we load the local eval JSONL (20 samples) and publish it as a Weave Dataset.
    """)
    return


@app.cell
def _(DATA_DIR, mo, pathlib, preview_table, weave, weave_client):
    import json

    mo.stop(weave_client is None)
    eval_data_path = DATA_DIR / "eval_data" / "documents.jsonl"
    mo.stop(
        not eval_data_path.exists(),
        mo.callout(
            mo.md(
                f"Missing eval dataset at `{eval_data_path}`. "
                "Place `documents.jsonl` under `rag-advanced/data/eval_data/` and re-run."
            ),
            kind="danger",
        ),
    )
    eval_rows = [
        json.loads(line)
        for line in eval_data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_dataset = weave.Dataset(name="eval_data", rows=eval_rows)
    _ref = weave.publish(eval_dataset)
    mo.vstack(
        [
            mo.md(
                f"**Number of evaluation samples:** {len(eval_dataset.rows)}  \n"
                f"Published Weave Dataset **eval_data**: `{_ref.uri()}`"
            ),
            preview_table(eval_dataset.rows, n=1),
        ]
    )
    return (eval_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Iterating through each sample is easy.

    We have the question, ground truth answer and ground truth contexts.
    """)
    return


@app.cell
def _(eval_dataset, mo):
    _sample = dict(eval_dataset.rows[0])
    mo.md(
        f"""**Sample question:** {_sample.get("question", "")}

    **Answer:** {_sample.get("answer", "")}

    **Contexts:** {len(_sample.get("contexts") or [])} ground-truth chunk(s)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating the Retriever

    The fundamental idea of evaluating a retriever is to check how well the retrieved content matches the expected contents. For evaluating a RAG pipeline end to end, we need query and ground truth answer pairs. The ground truth answer must be grounded on some "ground" truth chunks. This is a search problem, it's easiest to start with traditional Information retrieval metrics.

    You might already have access to such evaluation dataset depending on the nature of your application or you can synthetically build one. To build one you can retrieve random documents/chunks and ask an LLM to generate query-answer pairs - the underlying documents/chunks will act as your ground truth chunk.

    In the sections below, we will look at different metrics that can be used to evaluate the retriever we built in the last chapter.

    First let us download the chunked data (from chapter 1) and index it using a simple TFIDF based retriever.

    Pick where to fetch `chunked_data` from below. The **course dataset** option uses the public Weave ref published for this course. The **my project** option loads `chunked_data:latest` from your own Weave project (e.g. what you published in Chapter 1).
    """)
    return


@app.cell
def _(mo):
    data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (chunked_data:latest from your Weave project)": "own",
        },
        value="Course dataset (public Weave ref shared by the course)",
        label="Fetch chunked_data from:",
    )
    data_source
    return (data_source,)


@app.cell
def _(data_source, mo, preview_table, weave, weave_client):
    mo.stop(weave_client is None)
    _chunked_data_ref = (
        "weave:///wandb-smle/rag-course/object/chunked_data:CRbEh79BgxWPqbD1rOozLNw12pky0LhQYF2PIXgc2pQ"
        if data_source.value == "course"
        else "chunked_data:latest"
    )
    with mo.status.spinner(title="Fetching the `chunked_data` Dataset from Weave..."):
        chunked_data = weave.ref(_chunked_data_ref).get()
    mo.vstack(
        [
            mo.md(f"**Number of chunked rows:** {len(chunked_data.rows)}  \nRef: `{_chunked_data_ref}`"),
            preview_table(chunked_data.rows, n=2),
        ]
    )
    return (chunked_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will import the `TFIDFRetriever` which is an instance of `weave.Model` and index the chunked data from the last chapter.
    """)
    return


@app.cell
def _(chunked_data, mo, show_source):
    from scripts.retriever import TFIDFRetriever

    retriever = TFIDFRetriever()
    with mo.status.spinner(title="Indexing chunks with TF-IDF..."):
        retriever.index_data(list(map(dict, chunked_data.rows)))
    mo.vstack(
        [
            show_source(TFIDFRetriever),
            mo.callout(
                mo.md(f"Indexed **{len(chunked_data.rows)}** chunks into `TFIDFRetriever`."),
                kind="success",
            ),
        ]
    )
    return TFIDFRetriever, retriever


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics to evaluate retriever

    We can evaluate a retriever using traditional ML metrics. We can also evaluate it by using a powerful LLM (next section).

    Below we are importing both traditional metrics and LLM as a judge metric from the `scripts/retrieval_metrics.py` file.

    * **Hit Rate**: Measures the proportion of queries where the retriever successfully returns at least one relevant document.
    * **MRR (Mean Reciprocal Rank)**: Evaluates how quickly the retriever returns the first relevant document, based on the reciprocal of its rank.
    * **NDCG (Normalized Discounted Cumulative Gain)**: Assesses the quality of the ranked retrieval results, giving more importance to relevant documents appearing earlier.
    * **MAP (Mean Average Precision)**: Computes the mean precision across all relevant documents retrieved, considering the rank of each relevant document.
    * **Precision**: Measures the ratio of relevant documents retrieved to the total documents retrieved by the retriever.
    * **Recall**: Evaluates the ratio of relevant documents retrieved to the total relevant documents available for the query.
    * **F1 Score**: The harmonic mean of precision and recall, providing a balance between both metrics to gauge retriever performance.
    """)
    return


@app.cell
def _(show_source):
    from scripts.retrieval_metrics import IR_METRICS

    return (IR_METRICS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us first understand the basic traditional metrics we will be using. Each metric expects a `model_output` which is a list of retrieved chunks from the retriever and `contexts` which is a list of ground truth contexts.
    """)
    return


@app.cell
def _(IR_METRICS, mo, show_source):
    mo.accordion(
        {
            f"`{scorer.__name__}`": show_source(scorer)
            for scorer in IR_METRICS
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Evaluating retrieval on IR metrics

    Trial key note: this evaluation runs **sequentially** (`WEAVE_PARALLELISM=1`) over 20 samples. It is mostly local metric math (no Cohere), so it should finish relatively quickly.
    """)
    return


@app.cell
def _(mo):
    ir_eval_button = mo.ui.run_button(label="Run IR retrieval evaluation")
    ir_eval_button
    return (ir_eval_button,)


@app.cell
async def _(
    IR_METRICS,
    eval_dataset,
    ir_eval_button,
    mo,
    retriever,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not ir_eval_button.value,
        mo.md("*Click the button above to run the IR retrieval evaluation.*"),
    )
    retrieval_eval_ir = weave.Evaluation(
        name="Retrieval_Evaluation",
        dataset=eval_dataset,
        scorers=IR_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
    )
    with mo.status.spinner(title="Running IR retrieval evaluation..."):
        retrieval_scores_ir = await retrieval_eval_ir.evaluate(retriever)
    mo.md(f"```python\n{retrieval_scores_ir!r}\n```")
    return retrieval_eval_ir, retrieval_scores_ir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using an LLM evaluator for evaluating retriever

    **ref: https://arxiv.org/pdf/2406.06519**

    How do we evaluate if we don't have any ground truth?

    We can use a powerful LLM as a judge to evaluate the retriever.

    Trial key note: this calls Cohere for every sample — expect on the order of **~1–2 minutes** at 20 req/min for 20 samples.
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.retrieval_metrics import LLM_METRICS as RETRIEVAL_LLM_METRICS

    mo.accordion(
        {
            f"`{metric.__name__}`": show_source(metric)
            for metric in RETRIEVAL_LLM_METRICS
        },
        lazy=True,
    )
    return (RETRIEVAL_LLM_METRICS,)


@app.cell
def _(mo):
    retrieval_llm_button = mo.ui.run_button(
        label="Run LLM-judge retrieval evaluation (calls Cohere)"
    )
    retrieval_llm_button
    return (retrieval_llm_button,)


@app.cell
async def _(
    RETRIEVAL_LLM_METRICS,
    eval_dataset,
    mo,
    retrieval_llm_button,
    retriever,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not retrieval_llm_button.value,
        mo.md("*Click the button above to run the LLM-judge retrieval evaluation.*"),
    )
    retrieval_eval_llm = weave.Evaluation(
        name="LLM_Judge_Retrieval_Evaluation",
        dataset=eval_dataset,
        scorers=RETRIEVAL_LLM_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
    )
    with mo.status.spinner(title="Running LLM-judge retrieval evaluation (Cohere)..."):
        retrieval_scores_llm = await retrieval_eval_llm.evaluate(retriever)
    mo.md(f"```python\n{retrieval_scores_llm!r}\n```")
    return retrieval_eval_llm, retrieval_scores_llm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating the Response
    """)
    return


@app.cell
def _(NOTEBOOKS_DIR, mo, retriever, show_source):
    from scripts.rag_pipeline import SimpleRAGPipeline
    from scripts.response_generator import SimpleResponseGenerator

    INITIAL_PROMPT = (NOTEBOOKS_DIR / "prompts" / "initial_system.txt").read_text()
    response_generator = SimpleResponseGenerator(
        model="command-r-08-2024", prompt=INITIAL_PROMPT
    )
    rag_pipeline = SimpleRAGPipeline(
        retriever=retriever, response_generator=response_generator, top_k=5
    )
    mo.vstack(
        [
            show_source(SimpleRAGPipeline),
            mo.callout(
                mo.md(
                    "Initialized `SimpleRAGPipeline` with `command-r-08-2024` "
                    "and the Chapter 1 system prompt."
                ),
                kind="success",
            ),
        ]
    )
    return (
        INITIAL_PROMPT,
        SimpleRAGPipeline,
        SimpleResponseGenerator,
        rag_pipeline,
        response_generator,
    )


@app.cell
def _(mo, show_source):
    from scripts.response_metrics import NLP_METRICS

    mo.accordion(
        {f"`{scorer.__name__}`": show_source(scorer) for scorer in NLP_METRICS},
        lazy=True,
    )
    return (NLP_METRICS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Trial key note: response evaluation generates an answer per sample (Cohere chat).
    Expect on the order of **~1–2+ minutes** for 20 samples at 20 req/min.
    """)
    return


@app.cell
def _(mo):
    response_nlp_button = mo.ui.run_button(
        label="Run NLP response evaluation (calls Cohere)"
    )
    response_nlp_button
    return (response_nlp_button,)


@app.cell
async def _(
    NLP_METRICS,
    eval_dataset,
    mo,
    rag_pipeline,
    response_nlp_button,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not response_nlp_button.value,
        mo.md("*Click the button above to run the NLP response evaluation.*"),
    )
    response_eval_nlp = weave.Evaluation(
        name="Response_Evaluation",
        dataset=eval_dataset,
        scorers=NLP_METRICS[:-1],
        preprocess_model_input=lambda x: {"query": x["question"]},
    )
    with mo.status.spinner(title="Running NLP response evaluation (Cohere)..."):
        response_scores_nlp = await response_eval_nlp.evaluate(rag_pipeline)
    mo.md(f"```python\n{response_scores_nlp!r}\n```")
    return response_eval_nlp, response_scores_nlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using an LLM as a Response Judge

    Some metrics cannot be defined objectively and are particularly useful for more subjective or complex criteria.
    We care about correctness, faithfulness, and relevance.

    - **Answer Correctness** — Is the generated answer correct compared to the reference and thoroughly answers the user's query?
    - **Answer Relevancy** — Is the generated answer relevant and comprehensive?
    - **Answer Factfulness** — Is the generated answer factually consistent with the context document?

    Trial key note: correctness evaluation adds another Cohere call per sample on top of generation — budget your monthly quota.
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.response_metrics import LLM_METRICS as RESPONSE_LLM_METRICS

    mo.accordion(
        {
            f"`{metric.__name__}`": show_source(metric)
            for metric in RESPONSE_LLM_METRICS
        },
        lazy=True,
    )
    return (RESPONSE_LLM_METRICS,)


@app.cell
def _(mo):
    correctness_button = mo.ui.run_button(
        label="Run correctness LLM-judge evaluation (calls Cohere)"
    )
    correctness_button
    return (correctness_button,)


@app.cell
async def _(
    RESPONSE_LLM_METRICS,
    correctness_button,
    eval_dataset,
    mo,
    rag_pipeline,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not correctness_button.value,
        mo.md("*Click the button above to run the correctness evaluation.*"),
    )
    response_eval_correctness = weave.Evaluation(
        name="Correctness_Evaluation",
        dataset=eval_dataset,
        scorers=RESPONSE_LLM_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"]},
    )
    with mo.status.spinner(title="Running correctness evaluation (Cohere)..."):
        response_scores_correctness = await response_eval_correctness.evaluate(
            rag_pipeline
        )
    mo.md(f"```python\n{response_scores_correctness!r}\n```")
    return response_eval_correctness, response_scores_correctness


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.md(
        f"""
    Head over to [**your Weave evaluations / traces**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/evaluations)
    to inspect the runs above.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    ## Exercise

    1. Implement the `Relevance` and `Faithfulness` LLM evaluators (LLM as a Judge) and evaluate the pipeline on all the dimensions.
    2. Generate and share a W&B report with the following sections in the form of tables and charts:

        - Summary of the evaluation
        - Retrieval Evaluations
            - IR Metrics
            - LLM As a Retrieval Judge Metric
        - Response Evaluations
            - Traditional NLP Metrics
            - LLM Judgement Metrics
        - Overall Evaluations
        - Conclusion
    """),
        kind="neutral",
    )
    return


if __name__ == "__main__":
    app.run()
