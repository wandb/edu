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
    # Chapter 3: Data Ingestion and Preprocessing

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter03.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter03.py)

    <!--- @wandbcode{rag-course-03} -->

    At its core, RAG (Retrieval-Augmented Generation) is a method for integrating private data with pre-trained, instruction-tuned language models. The effectiveness of your RAG system is directly proportional to the quality of your data ingestion pipeline.

    Data ingestion encompasses both data sources and preprocessing. As with most machine learning systems, the principle of "garbage in, garbage out" applies to LLMs. Therefore, optimizing your data ingestion pipeline is crucial for RAG efficacy.

    Key considerations for efficient data ingestion:
    1. Periodic updates: Implement a system that can seamlessly update when data sources change.
    2. Quality control: Ensure data cleanliness and relevance.
    3. Scalability: Design the pipeline to handle increasing data volumes.

    **Tip**: When building your data ingestion pipeline, start with a small, representative sample of your data. This allows you to quickly iterate on your preprocessing steps and catch potential issues early. Focus on creating an end-to-end working system before optimizing specific components like chunk size, parsing strategies, or data formats (e.g., markdown, HTML, plain text).

    This is a [marimo](https://marimo.io) notebook: dependencies are declared inline and install automatically with
    `uvx marimo edit --sandbox Chapter03.py` or on [molab](https://molab.marimo.io).

    Cohere trial keys are rate-limited (**20 chat req/min**, **1,000 calls/month**). Evaluation cells are gated behind buttons; Weave concurrency stays at 1.
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
    return NOTEBOOKS_DIR, os


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
    import numpy as np
    import weave

    WANDB_PROJECT = "rag-course"
    weave_client = weave.init(WANDB_PROJECT)
    return np, weave, weave_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Best Practice**: Document your data preprocessing steps meticulously. This not only helps with reproducibility but also makes it easier to debug issues and optimize your pipeline later.

    ## Data Preparation

    We'll begin our journey with raw data processing.

    Our first step is to retrieve the most recent `raw_data` we logged into weave. We had logged this in Chapter 1 and we will use the same dataset as our starting point.
    """)
    return


@app.cell
def _(mo):
    raw_data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (raw_data:latest)": "own",
        },
        value="Course dataset (public Weave ref shared by the course)",
        label="Fetch raw_data from:",
    )
    raw_data_source
    return (raw_data_source,)


@app.cell
def _(mo, preview_table, raw_data_source, weave, weave_client):
    mo.stop(weave_client is None)
    _raw_ref = (
        "weave:///wandb-smle/rag-course/object/raw_data:YAG2pxOueKOaYIRkyG43s3xSsy5Blxmk62yrn9Mze60"
        if raw_data_source.value == "course"
        else "raw_data:latest"
    )
    with mo.status.spinner(title="Fetching `raw_data` from Weave..."):
        _raw_ds = weave.ref(_raw_ref).get()
    raw_data = list(map(dict, _raw_ds.rows[:]))
    mo.vstack(
        [
            mo.md(f"**Documents:** {len(raw_data)}  \nRef: `{_raw_ref}`"),
            preview_table(raw_data, n=2),
        ]
    )
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In chapter 1, we naively counted each word (as they appear in English text) as one token (`raw_tokens`). Now let's update to using the correct token counting strategy (`tokens`).

    We will use [Cohere's tokenizer](https://docs.cohere.com/docs/tokens-and-tokenizers) to calculate the accurate number of tokens per document in our `raw_data`. Both the correct token count and word count will be stored as metadata for each document.

    In RAG systems, accurate tokenization is crucial for proper text processing and context management. Let's examine two key functions: `tokenize_text` and `length_function`.

    **The `tokenize_text` Function**: This function tokenizes input text using Cohere's tokenization API. Here's how it works:

    1. It initializes a Cohere client using an API key stored in environment variables.
    2. It calls the `tokenize` method of the Cohere client, passing:
       - The input `text`
       - The specified `model` (defaulting to `command-r-08-2024`)
       - `offline=True` to use a locally cached tokenizer for efficiency

    The function returns a list of tokens, which are subword units that the model uses to process text.

    **The `length_function`**: This function calculates the number of tokens in a given text. It operates as follows:

    1. It calls `tokenize_text` to convert the input `text` into tokens.
    2. It returns the length of the resulting token list.

    Understanding the token count is essential because:
    - It helps determine if a text fits within a model's context window.
    - It allows for more accurate text chunking and processing in the RAG pipeline.
    - It provides a basis for estimating computational costs, as many API pricing models are based on token count.

    By using these functions, we ensure consistent and accurate tokenization throughout our RAG system, which is critical for retrieval accuracy and overall performance.
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.utils import (
        TOKENIZERS,
        get_special_tokens_set,
        length_function,
        tokenize_text,
    )

    mo.vstack([show_source(tokenize_text), show_source(length_function)])
    return TOKENIZERS, get_special_tokens_set, length_function


@app.cell
def _(length_function, mo, preview_table, raw_data):
    tokenized_raw_data = []
    with mo.status.spinner(title="Counting Cohere tokens for each document..."):
        for _doc in raw_data:
            _doc = dict(_doc)
            _doc["metadata"] = dict(_doc["metadata"])
            _words = _doc["metadata"].pop("raw_tokens", _doc["metadata"].get("words"))
            _doc["metadata"]["words"] = _words
            _doc["metadata"]["tokens"] = length_function(_doc["content"])
            tokenized_raw_data.append(_doc)
    preview_table(tokenized_raw_data, n=2)
    return (tokenized_raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see above, the `words` count (used in Chapter 1) is quite different from the actual `tokens` count. This discrepancy highlights the importance of accurate token counting in RAG systems. Knowing the correct number of tokens is crucial for several reasons:

    1. It helps decide whether to build a RAG pipeline or ingest the whole document into an LLM, especially now that many top LLMs support long context windows.
    2. It informs the optimal chunk size for efficient processing.
    3. It ensures efficient use of context windows and helps manage costs associated with token-based API calls.
    4. It aids in predicting and controlling response generation length, which is vital for maintaining coherent and relevant outputs.

    By accurately counting tokens, we can make informed decisions about our RAG system's architecture and optimize its performance and cost-effectiveness.

    **Tip**: Different tokenizers may produce slightly different results. Always use the same tokenizer that your target LLM uses to ensure consistency between preprocessing and model input.

    ## Preprocessing

    Now that we have our raw data prepared and correctly tokenized, the next crucial step is to pre-process this data. Preprocessing is essential for removing extraneous information and formatting that could interfere with our language model's understanding of the content.

    In this section, we'll focus on cleaning our data by removing markdown elements, special characters, and extra whitespace. This process will help streamline our text for more effective tokenization and ultimately improve the performance of our RAG system.

    Raw data often contains extra formatting information (like markdown elements) that, while useful for human readers, is not beneficial for LLMs. Removing these elements, along with special characters and extra whitespace, is essential in RAG preprocessing for several reasons:

    1. It eliminates noise and irrelevant information that could confuse the LLM.
    2. It ensures the model focuses solely on the content's semantic meaning.
    3. It standardizes input across various document types, creating a consistent format for the LLM.
    4. It can improve retrieval accuracy and response generation quality.

    To achieve this, we use two key functions:

    1. `convert_contents_to_text`: This function converts raw markdown to HTML, then uses BeautifulSoup to remove image links, images, and other formatting information.
    2. `make_text_tokenization_safe`: This function removes any special tokens present in the text. Special characters here are those defined in the tokenizer and may vary depending on the model used.

    By applying these preprocessing steps, we create clean, standardized input that's optimized for our LLM pipeline.

    **Best Practice**: When cleaning text data, be cautious about removing too much information. While it's important to remove noise, overzealous cleaning might inadvertently remove context that could be valuable for the LLM.
    """)
    return


@app.cell
def _(mo, show_source):
    from scripts.preprocess import convert_contents_to_text, make_text_tokenization_safe

    mo.vstack(
        [show_source(convert_contents_to_text), show_source(make_text_tokenization_safe)]
    )
    return convert_contents_to_text, make_text_tokenization_safe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are converting the raw markdown documents to text and making them tokenization-safe. This process involves removing special tokens that could interfere with the tokenization process. Let's examine the first 5 special tokens to understand what's being removed.

    After processing, you'll notice that the `parsed_tokens` count is smaller compared to the original `tokens` count. This reduction is expected and indicates that we've successfully removed extraneous formatting and special characters, resulting in a cleaner text representation that's more suitable for our LLM pipeline.

    This step is crucial for ensuring that our input data is optimized for tokenization and subsequent processing by the language model.

    This cell also **publishes** the result as a Weave Dataset named `parsed_data` in your project.
    """)
    return


@app.cell
def _(
    TOKENIZERS,
    convert_contents_to_text,
    get_special_tokens_set,
    length_function,
    make_text_tokenization_safe,
    mo,
    preview_table,
    tokenized_raw_data,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    special_tokens_set = get_special_tokens_set(TOKENIZERS["command-r"])
    parsed_data_local = []
    with mo.status.spinner(title="Parsing and cleaning documents..."):
        for _doc in tokenized_raw_data:
            _parsed_doc = dict(_doc)
            _parsed_doc["parsed_content"] = make_text_tokenization_safe(
                convert_contents_to_text(_doc["content"]),
                special_tokens_set=special_tokens_set,
            )
            _parsed_doc["metadata"] = dict(_doc["metadata"])
            _parsed_doc["metadata"]["parsed_tokens"] = length_function(
                _parsed_doc["parsed_content"]
            )
            parsed_data_local.append(_parsed_doc)

    parsed_dataset = weave.Dataset(name="parsed_data", rows=parsed_data_local)
    _ref = weave.publish(parsed_dataset)
    mo.vstack(
        [
            mo.md(
                f"**Special tokens (first 5):** `{list(special_tokens_set)[:5]}`  \n"
                f"**Parsed documents:** {len(parsed_data_local)}  \n"
                f"Published Weave Dataset **parsed_data**: `{_ref.uri()}`"
            ),
            preview_table(parsed_data_local, n=2),
        ]
    )
    return (parsed_data_local,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Chunking

    Next we demonstrate semantic chunking on a small sample (calls Cohere embeddings), then load the full `chunked_data` used for retrieval comparisons.

    Pick where to load `parsed_data` for the sample chunking step. The **course dataset** option uses the public Weave ref. The **my project** option uses the `parsed_data` you just published above (in memory).
    """)
    return


@app.cell
def _(mo):
    parsed_data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (parsed_data published above / parsed_data:latest)": "own",
        },
        value="Course dataset (public Weave ref shared by the course)",
        label="Fetch parsed_data from:",
    )
    parsed_data_source
    return (parsed_data_source,)


@app.cell
def _(
    mo,
    parsed_data_local,
    parsed_data_source,
    preview_table,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    if parsed_data_source.value == "course":
        _parsed_ref = (
            "weave:///wandb-smle/rag-course/object/parsed_data:62nddpeLJ7UY37VnwV44xHiRS3StDEQhpYnaX9E1NeI"
        )
        with mo.status.spinner(title="Fetching course `parsed_data` from Weave..."):
            parsed_data = list(map(dict, weave.ref(_parsed_ref).get().rows[:]))
        _label = _parsed_ref
    else:
        parsed_data = parsed_data_local
        _label = "in-memory (just published) / equivalent to parsed_data:latest"
    mo.vstack(
        [
            mo.md(f"**Parsed documents:** {len(parsed_data)}  \nSource: `{_label}`"),
            preview_table(parsed_data, n=2),
        ]
    )
    return (parsed_data,)


@app.cell
def _(show_source):
    from scripts.chunking import chunk_documents

    show_source(chunk_documents)
    return (chunk_documents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since we are doing semantic chunking, the chunking process can take a while. For now, let's just take the first 5 documents as a sample. This step **calls Cohere embeddings**.
    """)
    return


@app.cell
def _(mo):
    sample_chunk_button = mo.ui.run_button(
        label="Chunk first 5 docs (calls Cohere embeddings)"
    )
    sample_chunk_button
    return (sample_chunk_button,)


@app.cell
def _(chunk_documents, mo, parsed_data, preview_table, sample_chunk_button):
    mo.stop(
        not sample_chunk_button.value,
        mo.md("*Click the button above to run sample semantic chunking.*"),
    )
    with mo.status.spinner(title="Semantic-chunking 5 documents..."):
        sample_chunked_data = chunk_documents(parsed_data[:5])
    mo.vstack(
        [
            mo.md(f"**Sample chunks produced:** {len(sample_chunked_data)}"),
            preview_table(sample_chunked_data, n=2),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the rest of the data, we'll retrieve the chunked data from Weave.

    Pick the source below. The **course dataset** option uses the public `chunked_data` ref from Chapter 1. The **my project** option loads `chunked_data:latest` from your Weave project (e.g. after you publish your own chunks).
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
def _(chunked_data_source, mo, np, preview_table, weave, weave_client):
    mo.stop(weave_client is None)
    _chunked_ref = (
        "weave:///wandb-smle/rag-course/object/chunked_data:CRbEh79BgxWPqbD1rOozLNw12pky0LhQYF2PIXgc2pQ"
        if chunked_data_source.value == "course"
        else "chunked_data:latest"
    )
    with mo.status.spinner(title="Fetching `chunked_data` from Weave..."):
        _chunked_ds = weave.ref(_chunked_ref).get()
    chunked_data = list(map(dict, _chunked_ds.rows[:]))

    def _chunk_token_count(doc):
        meta = doc.get("metadata") or {}
        return (
            meta.get("parsed_tokens")
            or meta.get("cleaned_tokens")
            or meta.get("raw_tokens")
            or 0
        )

    _sizes = [_chunk_token_count(doc) for doc in chunked_data]
    mean_chunk_size = float(np.mean(_sizes)) if _sizes else 0.0
    std_chunk_size = float(np.std(_sizes)) if _sizes else 0.0
    mo.vstack(
        [
            mo.md(
                f"**Chunks:** {len(chunked_data)}  \n"
                f"Ref: `{_chunked_ref}`  \n"
                f"Mean chunk size: **{mean_chunk_size:.1f}**, Std: **{std_chunk_size:.1f}** "
                "(uses `parsed_tokens`, else `cleaned_tokens` / `raw_tokens`)"
            ),
            preview_table(chunked_data, n=2),
        ]
    )
    return (chunked_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tip**: Experiment with different chunking strategies and sizes. The optimal approach often depends on your specific use case and the nature of your documents. Monitor how changes in chunking affect both retrieval accuracy and LLM performance.

    ## Exploring Alternative Retrieval Methods

    Now that we have our data prepared, pre-processed, and chunked, it's time to focus on how we retrieve this information. While we've previously used TF-IDF as our baseline retrieval method, it's important to explore alternatives that might offer improved performance.

    In this section, we'll introduce the BM25 (Best Matching 25) retriever, a more sophisticated approach to information retrieval. BM25 is an evolution of TF-IDF that addresses some of its limitations. While TF-IDF simply weighs term frequency against document frequency, BM25 incorporates document length normalization and term frequency saturation. This means BM25 can better handle varying document lengths and prevents common terms from dominating the relevance score. As a result, BM25 often provides more nuanced rankings, especially for longer documents or queries with multiple terms.

    By comparing the performance of BM25 against our existing TF-IDF retriever, we can gain valuable insights into:

    1. The strengths and weaknesses of different retrieval algorithms in our specific use case
    2. The impact of more sophisticated ranking functions on RAG performance
    3. Potential areas for further optimization in our retrieval pipeline

    In our RAG pipeline, this could lead to more contextually relevant retrievals, potentially improving the quality of the generated responses. Let's implement the BM25 retriever and set up a comparative analysis with our TF-IDF baseline:
    """)
    return


@app.cell
def _(chunked_data, mo, show_source):
    from scripts.rag_pipeline import SimpleRAGPipeline
    from scripts.response_generator import SimpleResponseGenerator
    from scripts.retriever import BM25Retriever, TFIDFRetriever

    bm25_retriever = BM25Retriever()
    tfidf_retriever = TFIDFRetriever()
    with mo.status.spinner(title="Indexing BM25 and TF-IDF retrievers..."):
        bm25_retriever.index_data(chunked_data)
        tfidf_retriever.index_data(chunked_data)
    mo.vstack(
        [
            show_source(BM25Retriever),
            mo.callout(
                mo.md(
                    f"Indexed **{len(chunked_data)}** chunks into BM25 and TF-IDF retrievers."
                ),
                kind="success",
            ),
        ]
    )
    return (
        SimpleRAGPipeline,
        SimpleResponseGenerator,
        bm25_retriever,
        tfidf_retriever,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rest of the RAG pipeline remains unchanged. We'll use the same response generator and overall structure as before, allowing us to isolate the impact of our new retrieval method. This approach ensures a fair comparison between the TF-IDF and BM25 retrievers within our existing framework.
    """)
    return


@app.cell
def _(
    NOTEBOOKS_DIR,
    SimpleRAGPipeline,
    SimpleResponseGenerator,
    bm25_retriever,
    mo,
    tfidf_retriever,
):
    INITIAL_PROMPT = (NOTEBOOKS_DIR / "prompts" / "initial_system.txt").read_text()
    response_generator = SimpleResponseGenerator(
        model="command-r-08-2024", prompt=INITIAL_PROMPT
    )
    bm25_rag_pipeline = SimpleRAGPipeline(
        retriever=bm25_retriever, response_generator=response_generator, top_k=5
    )
    tfidf_rag_pipeline = SimpleRAGPipeline(
        retriever=tfidf_retriever, response_generator=response_generator, top_k=5
    )
    mo.callout(
        mo.md("Built BM25 and TF-IDF `SimpleRAGPipeline`s with `command-r-08-2024`."),
        kind="success",
    )
    return bm25_rag_pipeline, tfidf_rag_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate and compare the changes

    With our new retrieval method implemented, it's crucial to evaluate its performance and compare it to our baseline. This evaluation will help us understand the impact of our changes and determine whether the BM25 retriever offers improvements over the TF-IDF method.

    In this section, we'll use the evaluation dataset and metrics from Chapter 2 to assess both the retrieval performance and the overall RAG pipeline performance with each retriever.

    We are primarily interested in two aspects:
    1. The impact of preprocessing on retrieval metrics
    2. The effect of different retrieval methods on response metrics

    To address these points, we will evaluate:
    1. Retrieval metrics for both TF-IDF and BM25 retrievers
    2. Response metrics for the RAG pipeline using both retrievers

    We'll begin by fetching the evaluation dataset and metrics used in Chapter 2.
    """)
    return


@app.cell
def _():
    from scripts.response_metrics import ALL_METRICS as RESPONSE_METRICS
    from scripts.retrieval_metrics import ALL_METRICS as RETRIEVAL_METRICS

    return RESPONSE_METRICS, RETRIEVAL_METRICS


@app.cell
def _(mo):
    eval_data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (eval_data:latest / Dataset:latest)": "own",
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
    with mo.status.spinner(title="Fetching eval dataset from Weave..."):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Trial key note: retrieval IR metrics are cheap; LLM-judge parts inside `RETRIEVAL_METRICS` call Cohere. Expect several minutes for 20 samples at 20 req/min.
    """)
    return


@app.cell
def _(mo):
    retrieval_eval_button = mo.ui.run_button(
        label="Compare BM25 vs TF-IDF retrieval (may call Cohere)"
    )
    retrieval_eval_button
    return (retrieval_eval_button,)


@app.cell
async def _(
    RETRIEVAL_METRICS,
    bm25_retriever,
    eval_dataset,
    mo,
    retrieval_eval_button,
    tfidf_retriever,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not retrieval_eval_button.value,
        mo.md("*Click the button above to run retrieval evaluations.*"),
    )
    retrieval_evaluation = weave.Evaluation(
        name="Retrieval_Evaluation",
        dataset=eval_dataset,
        scorers=RETRIEVAL_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
    )
    with mo.status.spinner(title="Evaluating BM25 retriever..."):
        bm25_retrieval_scores = await retrieval_evaluation.evaluate(bm25_retriever)
    with mo.status.spinner(title="Evaluating TF-IDF retriever..."):
        tfidf_retrieval_scores = await retrieval_evaluation.evaluate(tfidf_retriever)
    mo.md(
        f"""**BM25 retrieval scores**

    ```python
    {bm25_retrieval_scores!r}
    ```

    **TF-IDF retrieval scores**

    ```python
    {tfidf_retrieval_scores!r}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![compare_retrievers](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/03_compare_retrievers.png)

    BM25 outperforms TFIDF in most relevance metrics:
    - Higher MAP, NDCG, and MRR indicate better ranking and relevance.
    - Improved precision and recall suggest a better balance in retrieving relevant documents.
    - Slightly better hit rate.

    TFIDF advantages:
    - Significantly lower latency, beneficial for time-sensitive applications.

    Trade-offs:
    - BM25 offers superior retrieval performance at the cost of higher latency.
    - TFIDF is faster but generally less accurate in retrieval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Trial key note: response evaluation generates answers with Cohere for each sample × each pipeline (BM25 and TF-IDF). This is the most expensive cell in the chapter — budget your monthly quota.
    """)
    return


@app.cell
def _(mo):
    response_eval_button = mo.ui.run_button(
        label="Compare BM25 vs TF-IDF responses (calls Cohere)"
    )
    response_eval_button
    return (response_eval_button,)


@app.cell
async def _(
    RESPONSE_METRICS,
    bm25_rag_pipeline,
    eval_dataset,
    mo,
    response_eval_button,
    tfidf_rag_pipeline,
    weave,
    weave_client,
):
    mo.stop(weave_client is None)
    mo.stop(
        not response_eval_button.value,
        mo.md("*Click the button above to run response evaluations.*"),
    )
    response_evaluations = weave.Evaluation(
        name="Response_Evaluation",
        dataset=eval_dataset,
        scorers=RESPONSE_METRICS,
        preprocess_model_input=lambda x: {"query": x["question"]},
    )
    with mo.status.spinner(title="Evaluating BM25 RAG responses (Cohere)..."):
        bm25_response_scores = await response_evaluations.evaluate(bm25_rag_pipeline)
    with mo.status.spinner(title="Evaluating TF-IDF RAG responses (Cohere)..."):
        tfidf_response_scores = await response_evaluations.evaluate(tfidf_rag_pipeline)
    mo.md(
        f"""**BM25 response scores**

    ```python
    {bm25_response_scores!r}
    ```

    **TF-IDF response scores**

    ```python
    {tfidf_response_scores!r}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.md(
        f"""
    ![compare_retriever_responses](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/03_compare_retriever_responses.png)

    Again BM25 retriever outperforms TFIDF across most response metrics (ROUGE, BLEU, Correctness, Response Score), leading to more relevant and accurate responses.

    - Slightly better ROUGE and BLEU scores indicate more overlap with reference responses.
    - Higher LLM Response Scorer results suggest more accurate and coherent responses.
    - Marginally higher Levenshtein and diff scores show slight differences from reference responses.

    TFIDF advantages:
      - Lower latency, making it more efficient for quicker response generation.

    Trade-offs:
      - BM25 offers superior response quality at the cost of higher latency.
      - TFIDF is faster but generally produces less accurate results.

    Choose based on your priority: response quality (BM25) or speed (TFIDF).

    Inspect runs in [**your Weave evaluations**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/evaluations).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    ## Key Takeaways

    1. Data Quality is Crucial: The effectiveness of a RAG system heavily depends on the quality of data ingestion and preprocessing. "Garbage in, garbage out" applies to LLMs as well.
    2. Accurate Tokenization: Using the correct tokenization strategy is essential for proper text processing and context management. The actual token count often differs significantly from simple word counts.
    3. Preprocessing Importance: Cleaning raw data by removing markdown elements, special characters, and extra whitespace is crucial for optimizing LLM input and improving retrieval accuracy.
    4. Semantic Chunking: This advanced chunking strategy groups similar sentences, preserving context and improving retrieval relevance compared to simpler methods like fixed-length splitting.
    5. Retrieval Method Comparison: Exploring alternative retrieval methods, such as BM25 vs. TF-IDF, can lead to significant improvements in RAG system performance.
    6. Evaluation is Key: Regularly evaluating both retrieval and response metrics is crucial for understanding the impact of changes and optimizing the RAG pipeline.
    7. Experimentation: The optimal approach for data preprocessing, chunking, and retrieval often depends on the specific use case. Experimentation is encouraged to find the best configuration for your RAG system.
    8. Scalability and Updates: When building a data ingestion pipeline, consider scalability and the ability to handle periodic updates as data sources change.

    ## Exercise

    1. Add more data sources to the RAG system. — Add Jupyter Notebooks from the wandb/examples repo.
    2. Use a different chunking method. — Try your own parsing and chunking method.
    3. Use a small-to-big retrieval method. Where we embed small documents but retrieve big documents → You can add the parent document to the metadata and modify the `Retriever.search` method.
    """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
