# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beautifulsoup4>=4.12.3",
#     "bm25s>=0.1.10",
#     "cohere>=5.13.6",
#     "fasttext-langdetect>=1.0.5",
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
    # Chapter 6: Response Synthesis and Prompting

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter06.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter06.py)

    <!--- @wandbcode{rag-course-06} -->

    Response synthesis is a critical component of RAG systems, responsible for generating coherent and accurate answers based on retrieved information. In this chapter, we'll explore techniques to improve response quality through iterative prompt engineering and model selection.

    Key concepts we'll cover:
    1. Baseline prompt evaluation
    2. Iterative prompt improvement
    3. Impact of model selection on response quality
    4. Comparative analysis of different prompting strategies

    This is a [marimo](https://marimo.io) notebook. Cohere trial keys are rate-limited (**20 chat req/min**, **Rerank 10 req/min**, **1,000 calls/month**). Hybrid indexing and each prompt-iteration evaluation are gated — expect heavy quota use if you run every eval.
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
    _PROMPT_FILES = [
        "initial_system.txt",
        "improved_prompt_v1.txt",
        "improved_prompt_v2.txt",
        "improved_prompt_v3.txt",
        "query_enhanced_system.txt",
    ]
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
    nltk.download("punkt_tab", quiet=True)
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
    import cohere
    import weave

    WANDB_PROJECT = "rag-course"
    weave_client = weave.init(WANDB_PROJECT)
    return WANDB_PROJECT, cohere, weave, weave_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data loading

    We'll start by loading the chunked data from Chapter 3. This chunked data will serve as the knowledge base for our RAG pipeline, allowing us to compare the effectiveness of our response synthesis techniques against a baseline system.
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
    Next, let's load the query enhancer, hybrid retriever, response generator and RAG pipeline from the previous chapters.
    """)
    return


@app.cell
def _(mo):
    from scripts.query_enhancer import QueryEnhancer
    from scripts.rag_pipeline import QueryEnhancedRAGPipeline
    from scripts.response_generator import QueryEnhanedResponseGenerator
    from scripts.retriever import HybridRetrieverReranker

    query_enhancer = QueryEnhancer()
    mo.callout(mo.md("`QueryEnhancer` initialized."), kind="success")
    return (
        HybridRetrieverReranker,
        QueryEnhancedRAGPipeline,
        QueryEnhanedResponseGenerator,
        query_enhancer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prompt iteration

    Prompt engineering is a crucial skill in developing effective RAG systems. We'll explore several iterations of prompt improvements:

    1. Baseline prompt
    2. Adding precise instructions
    3. Including response format examples
    4. Incorporating model reasoning

    For each iteration, we'll evaluate the impact on response quality using our established metrics.
    """)
    return


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
    from scripts.response_metrics import ALL_METRICS as RESPONSE_METRICS

    response_evaluations = weave.Evaluation(
        name="Response_Evaluation",
        dataset=eval_dataset,
        scorers=RESPONSE_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"]},
    )
    mo.vstack(
        [
            mo.md(
                f"**Evaluation samples:** {len(eval_dataset.rows)}  \nRef: `{_eval_ref}`"
            ),
            preview_table(eval_dataset.rows, n=2),
        ]
    )
    return RESPONSE_METRICS, eval_dataset, response_evaluations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Index the hybrid retriever (embeds all chunks via Cohere — large spend; also uses rerank later during eval at **10 req/min**).
    """)
    return


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
        mo.md("*Click the button above to index the hybrid retriever before running prompt evals.*"),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline Prompt Evaluation

    This simple prompt serves as our starting point. It provides basic instructions for the model to answer questions about W&B using only the provided context.

    **Tip**: When designing your initial prompt, aim for clarity and simplicity. However, be prepared to iterate and refine based on the results.

    **Best Practice**: Always establish a baseline performance to measure improvements against.
    """)
    return


@app.cell
def _(NOTEBOOKS_DIR, mo):
    INITIAL_PROMPT = (NOTEBOOKS_DIR / "prompts" / "initial_system.txt").read_text()
    mo.md(f"```text\n{INITIAL_PROMPT}\n```")
    return (INITIAL_PROMPT,)


@app.cell
def _(mo):
    baseline_eval_button = mo.ui.run_button(
        label="Evaluate baseline prompt (command-r-08-2024; heavy Cohere use)"
    )
    baseline_eval_button
    return (baseline_eval_button,)


@app.cell
async def _(
    INITIAL_PROMPT,
    QueryEnhancedRAGPipeline,
    QueryEnhanedResponseGenerator,
    baseline_eval_button,
    cohere,
    hybrid_retriever,
    mo,
    query_enhancer,
    response_evaluations,
):
    mo.stop(
        not baseline_eval_button.value,
        mo.md("*Click the button above to evaluate the baseline prompt.*"),
    )
    baseline_response_generator = QueryEnhanedResponseGenerator(
        model="command-r-08-2024",
        prompt=INITIAL_PROMPT,
        client=cohere.AsyncClientV2(),
    )

    class BaselineRAGPipeline(QueryEnhancedRAGPipeline):
        pass

    baseline_rag_pipeline = BaselineRAGPipeline(
        query_enhancer=query_enhancer,
        retriever=hybrid_retriever,
        response_generator=baseline_response_generator,
    )
    with mo.status.spinner(title="Evaluating baseline RAG pipeline..."):
        baseline_response_scores = await response_evaluations.evaluate(
            baseline_rag_pipeline
        )
    mo.md(f"```python\n{baseline_response_scores!r}\n```")
    return (
        BaselineRAGPipeline,
        baseline_rag_pipeline,
        baseline_response_generator,
        baseline_response_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Improved Prompt V1: Adding Precise Instructions

    We'll enhance the prompt by providing more detailed instructions: a clear role, dynamic language/intent, structured responses, markdown formatting, and edge-case handling.

    **Tip**: Adding specific instructions and defining the AI's role can significantly improve response quality.

    **Best Practice**: Include guidelines for handling edge cases, such as insufficient information or off-topic queries, in your prompt design.
    """)
    return


@app.cell
def _(NOTEBOOKS_DIR, mo):
    IMPROVED_PROMPT_V1 = (
        NOTEBOOKS_DIR / "prompts" / "improved_prompt_v1.txt"
    ).read_text()
    mo.md(f"```text\n{IMPROVED_PROMPT_V1}\n```")
    return (IMPROVED_PROMPT_V1,)


@app.cell
def _(mo):
    v1_eval_button = mo.ui.run_button(
        label="Evaluate improved prompt V1 (command-r-08-2024)"
    )
    v1_eval_button
    return (v1_eval_button,)


@app.cell
async def _(
    IMPROVED_PROMPT_V1,
    QueryEnhancedRAGPipeline,
    QueryEnhanedResponseGenerator,
    cohere,
    hybrid_retriever,
    mo,
    query_enhancer,
    response_evaluations,
    v1_eval_button,
):
    mo.stop(
        not v1_eval_button.value,
        mo.md("*Click the button above to evaluate improved prompt V1.*"),
    )
    improved_v1_response_generator = QueryEnhanedResponseGenerator(
        model="command-r-08-2024",
        prompt=IMPROVED_PROMPT_V1,
        client=cohere.AsyncClientV2(),
    )

    class ImprovedV1RAGPipeline(QueryEnhancedRAGPipeline):
        pass

    improved_v1_rag_pipeline = ImprovedV1RAGPipeline(
        query_enhancer=query_enhancer,
        retriever=hybrid_retriever,
        response_generator=improved_v1_response_generator,
    )
    with mo.status.spinner(title="Evaluating improved prompt V1..."):
        improved_v1_response_scores = await response_evaluations.evaluate(
            improved_v1_rag_pipeline
        )
    mo.md(f"```python\n{improved_v1_response_scores!r}\n```")
    return (
        ImprovedV1RAGPipeline,
        improved_v1_rag_pipeline,
        improved_v1_response_generator,
        improved_v1_response_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Improved Prompt V2: Including Response Format Examples

    This iteration incorporates a concrete example of a well-structured response (markdown, citations, detail level).

    **Tip**: Providing concrete examples in your prompt can help guide the model towards the desired output format and structure.

    **Best Practice**: When including examples, ensure they demonstrate key aspects like proper citation, use of markdown, and appropriate level of detail.
    """)
    return


@app.cell
def _(NOTEBOOKS_DIR, mo):
    IMPROVED_PROMPT_V2 = (
        NOTEBOOKS_DIR / "prompts" / "improved_prompt_v2.txt"
    ).read_text()
    mo.md(f"```text\n{IMPROVED_PROMPT_V2}\n```")
    return (IMPROVED_PROMPT_V2,)


@app.cell
def _(mo):
    v2_eval_button = mo.ui.run_button(
        label="Evaluate improved prompt V2 (command-r-08-2024)"
    )
    v2_eval_button
    return (v2_eval_button,)


@app.cell
async def _(
    IMPROVED_PROMPT_V2,
    QueryEnhancedRAGPipeline,
    QueryEnhanedResponseGenerator,
    cohere,
    hybrid_retriever,
    mo,
    query_enhancer,
    response_evaluations,
    v2_eval_button,
):
    mo.stop(
        not v2_eval_button.value,
        mo.md("*Click the button above to evaluate improved prompt V2.*"),
    )
    improved_v2_response_generator = QueryEnhanedResponseGenerator(
        model="command-r-08-2024",
        prompt=IMPROVED_PROMPT_V2,
        client=cohere.AsyncClientV2(),
    )

    class ImprovedV2RAGPipeline(QueryEnhancedRAGPipeline):
        pass

    improved_v2_rag_pipeline = ImprovedV2RAGPipeline(
        query_enhancer=query_enhancer,
        retriever=hybrid_retriever,
        response_generator=improved_v2_response_generator,
    )
    with mo.status.spinner(title="Evaluating improved prompt V2..."):
        improved_v2_response_scores = await response_evaluations.evaluate(
            improved_v2_rag_pipeline
        )
    mo.md(f"```python\n{improved_v2_response_scores!r}\n```")
    return (
        ImprovedV2RAGPipeline,
        improved_v2_rag_pipeline,
        improved_v2_response_generator,
        improved_v2_response_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Improved Prompt V3: Incorporating Model Reasoning

    This iteration focuses on structured reasoning and transparency — asking the model to explain its thought process step by step.

    **Tip**: Encouraging the model to explain its reasoning process can lead to more transparent and logical responses.

    **Best Practice**: Structure your prompt to guide the model through a step-by-step approach for complex queries.
    """)
    return


@app.cell
def _(NOTEBOOKS_DIR, mo):
    IMPROVED_PROMPT_V3 = (
        NOTEBOOKS_DIR / "prompts" / "improved_prompt_v3.txt"
    ).read_text()
    mo.md(f"```text\n{IMPROVED_PROMPT_V3}\n```")
    return (IMPROVED_PROMPT_V3,)


@app.cell
def _(mo):
    v3_eval_button = mo.ui.run_button(
        label="Evaluate improved prompt V3 (command-r-08-2024)"
    )
    v3_eval_button
    return (v3_eval_button,)


@app.cell
async def _(
    IMPROVED_PROMPT_V3,
    QueryEnhancedRAGPipeline,
    QueryEnhanedResponseGenerator,
    cohere,
    hybrid_retriever,
    mo,
    query_enhancer,
    response_evaluations,
    v3_eval_button,
):
    mo.stop(
        not v3_eval_button.value,
        mo.md("*Click the button above to evaluate improved prompt V3.*"),
    )
    improved_v3_response_generator = QueryEnhanedResponseGenerator(
        model="command-r-08-2024",
        prompt=IMPROVED_PROMPT_V3,
        client=cohere.AsyncClientV2(),
    )

    class ImprovedV3RAGPipeline(QueryEnhancedRAGPipeline):
        pass

    improved_v3_rag_pipeline = ImprovedV3RAGPipeline(
        query_enhancer=query_enhancer,
        retriever=hybrid_retriever,
        response_generator=improved_v3_response_generator,
    )
    with mo.status.spinner(title="Evaluating improved prompt V3..."):
        improved_v3_response_scores = await response_evaluations.evaluate(
            improved_v3_rag_pipeline
        )
    mo.md(f"```python\n{improved_v3_response_scores!r}\n```")
    return (
        ImprovedV3RAGPipeline,
        improved_v3_rag_pipeline,
        improved_v3_response_generator,
        improved_v3_response_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Improvement: Leveraging Advanced Language Models

    After iterating on our prompt engineering, we now take the next step by utilizing a more advanced language model (`command-r-plus-08-2024`, formerly the undated `command-r-plus` alias). Combining our refined prompt with a more sophisticated model aims to improve coherence, instruction following, and depth.

    **Tip**: Don't rely solely on prompt engineering; consider the capabilities of different models in your iterative improvement process.

    **Best Practice**: Balance the trade-off between response quality and latency based on your specific use-case requirements.
    """)
    return


@app.cell
def _(mo):
    v4_eval_button = mo.ui.run_button(
        label="Evaluate V3 prompt + command-r-plus-08-2024"
    )
    v4_eval_button
    return (v4_eval_button,)


@app.cell
async def _(
    IMPROVED_PROMPT_V3,
    QueryEnhancedRAGPipeline,
    QueryEnhanedResponseGenerator,
    cohere,
    hybrid_retriever,
    mo,
    query_enhancer,
    response_evaluations,
    v4_eval_button,
):
    mo.stop(
        not v4_eval_button.value,
        mo.md("*Click the button above to evaluate the model upgrade.*"),
    )
    improved_v4_response_generator = QueryEnhanedResponseGenerator(
        model="command-r-plus-08-2024",
        prompt=IMPROVED_PROMPT_V3,
        client=cohere.AsyncClientV2(),
    )

    class ImprovedV4RAGPipeline(QueryEnhancedRAGPipeline):
        pass

    improved_v4_rag_pipeline = ImprovedV4RAGPipeline(
        query_enhancer=query_enhancer,
        retriever=hybrid_retriever,
        response_generator=improved_v4_response_generator,
    )
    with mo.status.spinner(title="Evaluating V3 prompt + command-r-plus-08-2024..."):
        improved_v4_response_scores = await response_evaluations.evaluate(
            improved_v4_rag_pipeline
        )
    mo.md(f"```python\n{improved_v4_response_scores!r}\n```")
    return (
        ImprovedV4RAGPipeline,
        improved_v4_rag_pipeline,
        improved_v4_response_generator,
        improved_v4_response_scores,
    )


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.md(
        f"""
    ## Comparing Evaluations

    ![compare_retriever_responses](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/06_compare_prompts.png)

    ### Comparing RAG Pipeline Iterations

    1. **Response Quality Improvement**: Later prompt iterations significantly outperformed the baseline in LLM Response Scorer metrics.
    2. **Trade-off Between Quality and Latency**: Higher quality often means higher latency.
    3. **Incremental Gains**: Each iteration showed improvements across metrics.
    4. **Metric Variability**: Different metrics capture different aspects of performance.

    ### Learnings

    1. Prompt engineering can significantly impact response quality without changing the underlying model.
    2. Combining refined prompts with more advanced models (V4) can lead to synergistic improvements.
    3. Use multiple evaluation metrics for a comprehensive view.
    4. Balance response quality and latency for your use case.

    Inspect runs in [**your Weave evaluations**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/evaluations).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    ## Key Takeaways

    1. Iterative Prompt Engineering: Systematic refinement of prompts can significantly enhance response quality without changing the underlying model.
    2. Structured Instructions: Clear, detailed prompts with specific roles, formatting guidelines, and edge case handling improve response coherence and relevance.
    3. Example Integration: Including well-crafted examples in prompts helps guide the model towards desired output structure and content quality.
    4. Reasoning Transparency: Prompting the model to explain its thought process leads to more logical, comprehensive, and trustworthy responses.
    5. Model-Prompt Synergy: Combining refined prompts with more advanced language models (`command-r-08-2024` → `command-r-plus-08-2024`) can yield synergistic improvements.
    6. Performance Trade-offs: Higher quality responses often come at the cost of increased latency.
    7. Multifaceted Evaluation: Use a combination of metrics to comprehensively assess improvements.
    8. Continuous Optimization: RAG system development is an ongoing process.
    9. Scalability and Efficiency: As prompt complexity increases, consider the impact on system efficiency in real-world applications.
    """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
