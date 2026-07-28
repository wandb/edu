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
    # Chapter 4: Query Enhancement

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter04.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter04.py)

    <!--- @wandbcode{rag-course-04} -->

    Query enhancement is an intermediate step that uses LLMs to improve the quality of user queries. This can include:
    - Making queries grammatically correct
    - Breaking down complex queries into sub-queries
    - Extracting query intent
    - Augmenting queries with chat history
    - Extracting relevant keywords

    By working through this notebook, you will:

    1. Implement a Query Enhancement module that performs:
       - Language identification
       - Intent classification
       - Sub-query generation

    2. Integrate it into a RAG pipeline
    3. Compare and evaluate its impact against a baseline RAG system

    This is a [marimo](https://marimo.io) notebook: dependencies install automatically with
    `uvx marimo edit --sandbox Chapter04.py` or on [molab](https://molab.marimo.io).

    Cohere trial keys are rate-limited (**20 chat req/min**, **1,000 calls/month**). Paid cells are gated; Weave concurrency stays at 1. This chapter uses top-level `await` (no `nest_asyncio`).
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
    _PROMPT_FILES = ["initial_system.txt", "query_enhanced_system.txt"]
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initial steps:
    1. Log in to Weights & Biases (W&B)
    2. Configure environment variables for API access

    To obtain your Cohere API key, visit the [Cohere API dashboard](https://dashboard.cohere.com/api-keys).
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
    ## Data Loading

    We'll start by loading the chunked data from Chapter 3. As a reminder, semantic chunking is a technique that groups related sentences together, preserving context and improving retrieval relevance.

    This chunked data will serve as the knowledge base for our RAG pipeline, allowing us to compare the effectiveness of our query enhancement techniques against a baseline system.
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
    ## Query Enhancement

    In this section, we'll implement a query enhancement stage that improves our RAG pipeline. Our `QueryEnhancer` will perform three key tasks:

    1. **Language Identification**: Detect whether the query is in English, Japanese, or Korean. This allows us to provide responses in the user's preferred language, enhancing the user experience.

    2. **Intent Classification**: Determine if the query is relevant to our documentation. This helps filter out off-topic questions and allows for more appropriate responses.

    3. **Sub-query Generation**: Break down complex queries into smaller, more focused sub-queries. This improves retrieval by capturing different aspects of the original question.

    These enhancements serve two primary purposes:
    - Inform the response generator, allowing it to tailor its output based on language and intent.
    - Improve the retrieval process by using more targeted sub-queries.

    Let's implement our `QueryEnhancer` and see it in action:
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.query_enhancer import QueryEnhancer

    query_enhancer = QueryEnhancer()
    mo.vstack(
        [
            show_source(QueryEnhancer),
            mo.callout(mo.md("`QueryEnhancer` initialized."), kind="success"),
        ]
    )
    return QueryEnhancer, query_enhancer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyzing Query Enhancement Results

    Let's examine the output of our `QueryEnhancer` for the input: "How do I log images in lightning with wandb?"

    This step **calls Cohere**.
    """)
    return


@app.cell
def _(mo):
    enhance_button = mo.ui.run_button(
        label="Run QueryEnhancer sample (calls Cohere)"
    )
    enhance_button
    return (enhance_button,)


@app.cell
async def _(enhance_button, mo, query_enhancer):
    mo.stop(
        not enhance_button.value,
        mo.md("*Click the button above to enhance the sample query.*"),
    )
    sample_query = "How do I log images in lightning with wandb?"
    with mo.status.spinner(title="Enhancing query..."):
        enhance_response = await query_enhancer.predict(sample_query)
    mo.md(f"**Query:** {sample_query}\n\n```python\n{enhance_response!r}\n```")
    return enhance_response, sample_query


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our `QueryEnhancer` has provided three key pieces of information:

    1. **Language Detection**: The query is identified as English ('en'). This allows our system to respond in the appropriate language.

    2. **Sub-query Generation**: The original query is broken down into more specific sub-queries. These sub-queries help capture different aspects of the original question, potentially improving retrieval accuracy.

    3. **Intent Classification**: The query is classified under an intent such as "integrations". This suggests the user is asking about a specific integration between Lightning and Weights & Biases.

    By leveraging this enhanced query information, our RAG system can now perform more targeted retrieval and generate more relevant, context-aware responses.

    ## Retriever: Leveraging BM25 from Previous Chapter

    In our previous chapter, we explored the BM25 (Best Matching 25) retriever as an improvement over the basic TF-IDF approach. BM25 offers more nuanced rankings by incorporating document length normalization and term frequency saturation.

    For this notebook, we'll continue using the same BM25 retriever. This consistency serves two important purposes:

    1. **Fair Comparison**: By keeping the retriever constant, we can isolate the impact of our query enhancement techniques.
    2. **Leveraging Sub-queries**: Our new query-enhanced pipeline can take advantage of the sub-queries generated by the QueryEnhancer.
    """)
    return


@app.cell
def _(chunked_data, mo, show_source):
    from scripts.retriever import BM25Retriever

    retriever = BM25Retriever()
    with mo.status.spinner(title="Indexing BM25 retriever..."):
        retriever.index_data(chunked_data)
    mo.vstack(
        [
            show_source(BM25Retriever),
            mo.callout(
                mo.md(f"Indexed **{len(chunked_data)}** chunks into BM25."),
                kind="success",
            ),
        ]
    )
    return BM25Retriever, retriever


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Query-Enhanced Response Generation

    With the additional information extracted from our query - specifically the language and intent - we can now create a more sophisticated response generator. The `QueryEnhanedResponseGenerator` class leverages this enriched context to produce more tailored and relevant responses.

    Key Enhancements:
    1. **Language-Aware Responses**: By incorporating the detected language, we can adjust the response style and potentially use language-specific resources or examples.
    2. **Intent-Driven Generation**: The classified intent helps the model understand the user's goal, allowing for more focused and appropriate responses.
    3. **Dynamic Prompt Engineering**: The system prompt is dynamically formatted with language and intent information, guiding the LLM's response generation process.
    """)
    return


@app.cell
def _(show_source):
    from scripts.response_generator import QueryEnhanedResponseGenerator

    show_source(QueryEnhanedResponseGenerator)
    return (QueryEnhanedResponseGenerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the formatting of the system prompt with the detected language and intents. This crucial step allows us to dynamically adapt our instructions to the LLM based on the specific query context.

    ## Query-Enhanced RAG Pipeline

    The `QueryEnhancedRAGPipeline` takes full advantage of our query enhancements:

    1. **Multi-Query Retrieval**: Leverages sub-queries for multiple retrieval operations.
    2. **Context Deduplication**: Deduplicates retrieved chunks to optimize LLM input.
    3. **Intent-Based Flow Control**: Bypasses retrieval for off-topic / inappropriate intents.
    """)
    return


@app.cell
def _(show_source):
    from scripts.rag_pipeline import QueryEnhancedRAGPipeline

    show_source(QueryEnhancedRAGPipeline)
    return (QueryEnhancedRAGPipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting It All Together

    We'll initialize the response generator with our new prompt, set up the complete RAG pipeline, and test it with a sample query.
    """)
    return


@app.cell
def _(
    NOTEBOOKS_DIR,
    QueryEnhancedRAGPipeline,
    QueryEnhanedResponseGenerator,
    cohere,
    mo,
    query_enhancer,
    retriever,
):
    QUERY_ENHANCED_PROMPT = (
        NOTEBOOKS_DIR / "prompts" / "query_enhanced_system.txt"
    ).read_text()
    enhanced_response_generator = QueryEnhanedResponseGenerator(
        model="command-r-08-2024",
        prompt=QUERY_ENHANCED_PROMPT,
        client=cohere.AsyncClientV2(),
    )
    query_enhanced_rag_pipeline = QueryEnhancedRAGPipeline(
        query_enhancer=query_enhancer,
        retriever=retriever,
        response_generator=enhanced_response_generator,
        top_k=2,
    )
    mo.callout(
        mo.md(
            "Initialized `QueryEnhancedRAGPipeline` with `command-r-08-2024` "
            "and `query_enhanced_system.txt`."
        ),
        kind="success",
    )
    return (
        QUERY_ENHANCED_PROMPT,
        enhanced_response_generator,
        query_enhanced_rag_pipeline,
    )


@app.cell
def _(mo):
    pipeline_demo_button = mo.ui.run_button(
        label="Run query-enhanced RAG sample (calls Cohere)"
    )
    pipeline_demo_button
    return (pipeline_demo_button,)


@app.cell
async def _(mo, pipeline_demo_button, query_enhanced_rag_pipeline):
    mo.stop(
        not pipeline_demo_button.value,
        mo.md("*Click the button above to run the sample query through the enhanced pipeline.*"),
    )
    _demo_query = "How do I log images in lightning with wandb?"
    with mo.status.spinner(title="Running query-enhanced RAG pipeline..."):
        demo_response = await query_enhanced_rag_pipeline.predict(_demo_query)
    mo.md(f"**Query:** {_demo_query}\n\n{demo_response}")
    return (demo_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating Query Enhancement: Comparing Performance

    Now that we've implemented our query-enhanced RAG system, it's crucial to quantify its performance improvements over our baseline.

    In this section, we'll:

    1. Use the evaluation dataset from previous chapters to ensure consistency.
    2. Focus on response quality metrics, as both systems use the same BM25 retriever.
    3. Employ LLM-based metrics to assess the quality and relevance of generated responses.
    4. Compare `QueryEnhancedRAGPipeline` against `SimpleRAGPipeline`.

    Trial key note: this evaluation runs **two** full Cohere-backed pipelines over 20 samples — expect a long wait and heavy quota use.
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
    mo.vstack(
        [
            mo.md(
                f"**Evaluation samples:** {len(eval_dataset.rows)}  \nRef: `{_eval_ref}`"
            ),
            preview_table(eval_dataset.rows, n=2),
        ]
    )
    return (eval_dataset,)


@app.cell
def _(NOTEBOOKS_DIR, mo, retriever, show_source):
    from scripts.rag_pipeline import SimpleRAGPipeline
    from scripts.response_generator import SimpleResponseGenerator

    INITIAL_PROMPT = (NOTEBOOKS_DIR / "prompts" / "initial_system.txt").read_text()
    baseline_response_generator = SimpleResponseGenerator(
        model="command-r-08-2024", prompt=INITIAL_PROMPT
    )
    simple_rag_pipeline = SimpleRAGPipeline(
        retriever=retriever,
        response_generator=baseline_response_generator,
        top_k=5,
    )
    mo.vstack(
        [
            show_source(SimpleRAGPipeline),
            mo.callout(
                mo.md("Baseline `SimpleRAGPipeline` ready for comparison."),
                kind="success",
            ),
        ]
    )
    return (
        INITIAL_PROMPT,
        SimpleRAGPipeline,
        SimpleResponseGenerator,
        baseline_response_generator,
        simple_rag_pipeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this evaluation, we're primarily interested in assessing the overall response quality rather than individual retrieval metrics. Here's why:

    1. **Common Retriever**: Both pipelines use the same BM25 retriever, making retrieval metric comparisons less informative.
    2. **Enhanced Retrieval Process**: The query-enhanced pipeline retrieves more chunks due to its use of sub-queries, making direct retrieval comparisons potentially misleading.
    3. **Holistic Evaluation**: Our goal is to understand the end-to-end performance improvement, which is best captured by analyzing the final output quality.
    """)
    return


@app.cell
def _(mo):
    compare_eval_button = mo.ui.run_button(
        label="Compare simple vs query-enhanced responses (calls Cohere heavily)"
    )
    compare_eval_button
    return (compare_eval_button,)


@app.cell
async def _(
    compare_eval_button,
    eval_dataset,
    mo,
    query_enhanced_rag_pipeline,
    simple_rag_pipeline,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not compare_eval_button.value,
        mo.md("*Click the button above to run the comparative response evaluation.*"),
    )
    from scripts.response_metrics import LLM_METRICS as RESPONSE_LLM_METRICS

    response_evaluations = weave.Evaluation(
        name="Response_Evaluation",
        dataset=eval_dataset,
        scorers=RESPONSE_LLM_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"]},
    )
    with mo.status.spinner(title="Evaluating baseline SimpleRAGPipeline..."):
        baseline_response_scores = await response_evaluations.evaluate(
            simple_rag_pipeline
        )
    with mo.status.spinner(title="Evaluating QueryEnhancedRAGPipeline..."):
        query_enhanced_response_scores = await response_evaluations.evaluate(
            query_enhanced_rag_pipeline
        )
    mo.md(
        f"""**Baseline (SimpleRAG) scores**

```python
{baseline_response_scores!r}
```

**Query-enhanced scores**

```python
{query_enhanced_response_scores!r}
```
"""
    )
    return (
        RESPONSE_LLM_METRICS,
        baseline_response_scores,
        query_enhanced_response_scores,
        response_evaluations,
    )


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.md(
        f"""
    ![compare_retriever_responses](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/04_compare_query_enhanced_responses.png)

    ## Evaluation Results: Simple vs. Query-Enhanced RAG Pipelines

    ### Key Findings:

    1. **Response Quality**:
       - Both pipelines achieved similar correctness scores.
       - The SimpleRAGPipeline showed a slightly higher overall response quality score (0.75 vs 0.70).

    2. **Latency**:
       - The QueryEnhancedRAGPipeline exhibited significantly higher latency (21.40s vs 5.74s).
       - This increased latency is likely due to the additional processing in query enhancement.

    3. **Detailed Analysis**:
       - A careful examination of individual responses reveals that the QueryEnhancedRAGPipeline often generates more relevant and coherent answers.
       - However, these responses tend to be more verbose, which may have led to lower scores from the LLM Judge, which seems to favor conciseness.

    ### Takeaways:

    1. **Complexity vs. Performance**: Adding sophisticated features like query enhancement doesn't always lead to immediate improvements in automated metrics.
    2. **Latency Considerations**: In real-world applications, the significant increase in latency could impact user experience.
    3. **Metric Limitations**: The discrepancy between perceived quality and metric scores underscores the limitations of current evaluation methods.
    4. **Dataset Dependency**: These observations are specific to our current dataset.
    5. **Qualitative vs. Quantitative Analysis**: Combine automated metrics with qualitative analysis of individual responses.
    6. **Iterative Improvement**: Use results as a foundation for further refinement.

    Inspect runs in [**your Weave evaluations**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/evaluations).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    ## Key Takeaways

    1. Query Enhancement Complexity: Advanced techniques like language identification, intent classification, and sub-query generation can significantly improve RAG system sophistication, but also introduce complexity.
    2. Context-Aware Retrieval: Leveraging sub-queries and intent classification enables more comprehensive and targeted information retrieval.
    3. Safety and Robustness: Intent-based flow control enhances system safety by appropriately handling off-topic or inappropriate queries.
    4. Performance Trade-offs: Query enhancement can lead to more relevant responses but may increase latency.
    5. Evaluation Challenges: Standard metrics may not fully capture nuanced improvements in response quality.
    6. Iterative Development: RAG system optimization is an ongoing process.
    7. Holistic System Design: Effective RAG systems require careful integration of various components.
    8. Real-world Application: Balance theoretical improvements with practical concerns like user experience and system complexity.
    """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
