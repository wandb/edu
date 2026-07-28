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
    # Chapter 1

    [Open in molab](https://molab.marimo.io/github/wandb/edu/blob/main/rag-advanced/marimo/Chapter01.py) · [View source on GitHub](https://github.com/wandb/edu/blob/main/rag-advanced/marimo/Chapter01.py)

    <!--- @wandbcode{rag-course-01} -->

    In this chapter, we will learn about building a simple RAG pipeline. We will mainly focus on how to preprocess and chunk the data followed by building a simple retrieval engine without using any fancy "Vector Index". The idea is to show the inner working of a retrieval pipeline and make you understand the workflow from a user query to a generated response using an LLM.

    For this chapter you will need a Cohere API key and a Weights & Biases account.
    This is a [marimo](https://marimo.io) notebook: its dependencies are declared inline
    at the top of the file and are installed automatically when you open it with
    `uvx marimo edit --sandbox Chapter01.py` or on [molab](https://molab.marimo.io).
    """)
    return


@app.cell
def _(mo):
    import os
    import pathlib
    import sys
    import urllib.request

    # The course's shared code lives in ../notebooks/scripts; make it importable.
    NOTEBOOKS_DIR = pathlib.Path(mo.notebook_dir()).resolve().parent / "notebooks"
    DATA_DIR = pathlib.Path(mo.notebook_dir()).resolve().parent / "data"

    # When the notebook runs outside the repo (e.g. opened on molab, which only
    # imports this single file), fetch the shared course code from GitHub.
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
            {f"Source of `{symbol.__name__}`": mo.md(f"```python\n{source}\n```")}
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
    Now we are ready to start the chapter.

    We will start by logging into Weights & Biases and setup the API keys in the environment. You should be able to get your Cohere API key from [here](https://dashboard.cohere.com/api-keys) — signing up gives you a free **trial key** (rate-limited to 20 chat requests/minute and 1,000 API calls/month, which is enough for this course).
    """)
    return


@app.cell
def _(mo, os):
    def _saved_wandb_api_key():
        """Return a W&B API key from ~/.netrc, if the user is already logged in."""
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
    # Cohere's free trial keys allow only 20 chat requests/minute; keep Weave's
    # evaluation concurrency at 1 so parallel submissions don't hit 429 errors.
    os.environ.setdefault("WEAVE_PARALLELISM", "1")
    if wandb_key_input.value:
        os.environ["WANDB_API_KEY"] = wandb_key_input.value
    api_keys_ready = True
    return (api_keys_ready,)


@app.cell
def _():
    from typing import List

    import wandb
    import weave

    return List, wandb, weave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we will start a Weights & Biases run. We will be using this to download a [W&B Artifact](https://docs.wandb.ai/guides/artifacts) called `wandb_docs`. This is the raw Weights & Biases documentation. W&B Artifacts is suited for versiong different data sources which needs preprocessing/cleaning.
    """)
    return


@app.cell
def _(api_keys_ready, mo, wandb):
    mo.stop(not api_keys_ready)
    WANDB_PROJECT = "rag-course"
    run = wandb.init(project=WANDB_PROJECT, group="Chapter 1")
    return WANDB_PROJECT, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data ingestion

    ### Loading the data

    Use [W&B Artifacts](https://docs.wandb.ai/guides/artifacts) to track and version data as the inputs and outputs of your W&B Runs. For example, a model training run might take in a dataset as input and produce a trained model as output. W&B Artifacts is a powerful object storage with rich UI functionalities.

    Below we are downloading an artifact named `wandb_docs` which will download 400 odd markdown files in your `../data/wandb_docs` directory. This is will be our data source.
    """)
    return


@app.cell
def _(DATA_DIR, mo, pathlib, run):
    with mo.status.spinner(title="Downloading the `wandb_docs` artifact..."):
        documents_artifact = run.use_artifact(
            "wandb-smle/rag-course/wandb_docs:latest", type="dataset"
        )
        docs_dir = pathlib.Path(documents_artifact.download(str(DATA_DIR / "wandb_docs")))
    return (docs_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's inspect the `../data/wandb_docs` directory and look at the names of the first 5 files. We should see that they are all in markdown (`.md`) file format.
    """)
    return


@app.cell
def _(docs_dir, mo):
    docs_files = sorted(docs_dir.rglob("*.md"))
    mo.md(
        f"**Number of files:** {len(docs_files)}\n\n**First 5 files:**\n\n"
        + "\n".join(f"- `{f.relative_to(docs_dir)}`" for f in docs_files[:5])
    )
    return (docs_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets look at an example file. We can take the first element of the list (`docs_files`) and use the `Path.read_text` method to get the decoded contents of the file as a string.
    """)
    return


@app.cell
def _(docs_files, mo):
    mo.accordion(
        {f"`{docs_files[0].name}` (raw markdown)": mo.plain_text(docs_files[0].read_text())}
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    💡 Looking at the example, we see some structure and format to it.

    It is always a good practice to look through few examples to see if there is any pattern to your data source.
    It helps to come up with better preprocessing steps and chunking strategies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's store our documents as dictionaries with content (raw text) and metadata.

    Metadata is extra information for that data point which can be used to group together similar data points, or filter out a few data points.
    We will see in future chapters the importance of metadata and why it should not be ignored while building the ingestion pipeline.

    The metadata can be derived (`raw_tokens`) or is inherent (`source`) to the data point.

    Note that we are simply doing word counting and calling it `raw_tokens`.
    In practice we would be using a [tokenizer](https://docs.cohere.com/docs/tokens-and-tokenizers) to calculate the token counts but this naive calculation is an okay approximation for now.
    """)
    return


@app.cell
def _(docs_dir, docs_files, preview_table):
    # We'll store the files as dictionaries with some content and metadata
    data = []
    for _file in docs_files:
        _content = _file.read_text()
        data.append(
            {
                "content": _content,
                "metadata": {
                    "source": str(_file.relative_to(docs_dir)),
                    "raw_tokens": len(_content.split()),
                },
            }
        )
    preview_table(data, n=2)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Checking the total number of tokens of your data source is a good practice. In this case, the total tokens is more than 200k. Surely, most LLM providers cannot process this many tokens. Building a RAG is justified in such cases.
    """)
    return


@app.cell
def _(data, mo):
    total_tokens = sum(map(lambda x: x["metadata"]["raw_tokens"], data))
    mo.stat(
        value=f"{total_tokens:,}",
        label="Total tokens in dataset",
        caption="far beyond what most LLMs can process in one call",
        bordered=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## W&B Weave

    In the previous section we used W&B Artifacts to download the source documents.

    We could have used it to log our processed data but instead we will use W&B Weave for the task.

    Why?

    - W&B Weave is standalone and doesn't need backward compatibility with core Weights & Biases offerings.
    - W&B Weave is designed for modern LLMOps use case.
    - We would like to keep the data, "models" (function, API, collection of parser, extrators, and more) and evaluators in "one single source of truth" managed by W&B Weave.

    What?

    W&B Weave is a lightweight toolkit for tracking and evaluating LLM applications:

    - Log and debug language model inputs, outputs, and traces
    - Build rigorous, apples-to-apples evaluations for language model use cases
    - Organize all the information generated across the LLM workflow, from experimentation to evaluations to production

    The newly created list of dictionaries with `content` and `metadata` will now be logged as a W&B Weave Dataset called `raw_data`. Notice that our processed data is a list of dicts.

    Let's initialize W&B Weave. Once intialized, we will start tracking (more on it later) the inputs and the outputs along with underlying attributes (model name, top_k, etc.).
    """)
    return


@app.cell
def _(WANDB_PROJECT, api_keys_ready, mo, weave):
    mo.stop(not api_keys_ready)
    weave_client = weave.init(WANDB_PROJECT)
    return (weave_client,)


@app.cell
def _(data, mo, weave, weave_client):
    mo.stop(weave_client is None)
    # build weave dataset
    raw_data = weave.Dataset(name="raw_data", rows=data)

    # publish the dataset
    _ref = weave.publish(raw_data)
    mo.md(f"Published Weave Dataset **raw_data**: `{_ref.uri()}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![raw_data](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/01_raw_data.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chunking the data

    Each document contains a large number of tokens, so we need to split it into smaller chunks to manage the number of tokens per chunk. This approach serves three main purposes:

    * Most embedding models have a limit of 512 tokens per input (based on their training data and parameters).

    * Chunking allows us to retrieve and send only the most relevant portions to our LLM, significantly reducing the total token count. This helps keep the LLM's cost and processing time manageable.

    * When the text is small-sized, embedding models tend to generate better vectors as they can capture more fine-grained details and nuances in the text, resulting in more accurate representations.

    When choosing chunk size, consider these trade-offs:

    - Smaller chunks (100-200 tokens):
      * More precise retrieval
      * Better for finding specific details
      * May lack broader context

    - Larger chunks (500-1000 tokens):
      * Provide more context
      * Capture more coherent ideas
      * May introduce noise and reduce precision

    The optimal size depends on your data, expected queries, and model capabilities. Experiment with different sizes to find the best balance for your use case.

    Here we are chunking each content (text) to a maximum length of 300 tokens (`CHUNK_SIZE`).
    For now, we will not be overlapping (`CHUNK_OVERLAP`) the content of one chunk with another chunk.
    """)
    return


@app.cell
def _(List):
    # These are hyperparameters of our ingestion pipeline

    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 0

    def split_into_chunks(
        text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """Function to split the text into chunks of a maximum number of tokens
        ensure that the chunks are of size CHUNK_SIZE and overlap by chunk_overlap tokens
        use the `tokenizer.encode` method to tokenize the text
        """
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = tokens[start:end]
            chunks.append(" ".join(chunk))
            start = end - chunk_overlap
        return chunks

    return (split_into_chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use the `raw_data` artifact we just published as W&B Weave Dataset as the input to the chunking function.

    Pick where to fetch the datasets from below. The **course dataset** option resolves a full Weave ref URI — refs are public links, so the datasets the course published can be loaded by anyone, and the ones you publish can be shared the same way. The **my project** option loads the datasets you just published above from your own project. Switching the option re-runs every cell that depends on the data.
    """)
    return


@app.cell
def _(mo):
    data_source = mo.ui.radio(
        options={
            "Course dataset (public Weave ref shared by the course)": "course",
            "My project (the datasets you published above)": "own",
        },
        value="Course dataset (public Weave ref shared by the course)",
        label="Fetch Weave Datasets from:",
    )
    data_source
    return (data_source,)


@app.cell
def _(data_source, mo, preview_table, weave, weave_client):
    mo.stop(weave_client is None)
    _raw_data_ref = (
        "weave:///wandb-smle/rag-course/object/raw_data:YAG2pxOueKOaYIRkyG43s3xSsy5Blxmk62yrn9Mze60"
        if data_source.value == "course"
        else "raw_data:latest"
    )
    with mo.status.spinner(title="Fetching the `raw_data` Dataset from Weave..."):
        raw_data_ds = weave.ref(_raw_data_ref).get()
    # this is how we index into the data
    preview_table(raw_data_ds.rows, n=2)
    return (raw_data_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A Weave Dataset is compatible with the rest of the Weave workflow and is pythonic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us chunk each document in the raw data Artifact. We create a new list of dictionaries with the chuked text (`content`) and with `metadata`.
    """)
    return


@app.cell
def _(preview_table, raw_data_ds, split_into_chunks):
    chunked_data = []
    for _doc in raw_data_ds.rows:
        _chunks = split_into_chunks(_doc["content"])
        for _chunk in _chunks:
            chunked_data.append(
                {
                    "content": _chunk,
                    "metadata": {
                        "source": _doc["metadata"]["source"],
                        "raw_tokens": len(_chunk.split()),
                    },
                }
            )
    preview_table(chunked_data, n=2)
    return (chunked_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cleaning the data

    Cleaning up data is crucial for most AI pipelines and applies to a RAG/Agentic pipeline as well.
    Usually, higher quality chunks provided to an LLM generates a higher quality response.

    here, it is particularly important for several reasons:

    - **Tokenization issues**: Special tokens can interfere with the model's tokenization process, leading to unexpected interpretations of the text.
    - **Vocabulary pollution**: Unusual tokens can inflate the model's vocabulary, potentially diluting the importance of meaningful terms.
    - **Semantic distortion**: Special characters or formatting tokens (e.g., HTML tags) can alter the semantic meaning of sentences if not properly handled.
    - **Consistency**: Removing or standardizing special tokens ensures consistent representation across different data sources.
    - **Model efficiency**: Cleaner data often leads to more efficient model training and inference, as the model doesn't need to process irrelevant tokens.

    By cleaning the data, we ensure that our model focuses on the meaningful content rather than artifacts or noise in the text.

    We will use a simple function `make_text_tokenization_safe` to remove special tokens from the text.
    This is a good practice as most LLM providers do not like special tokens in the text.
    """)
    return


@app.cell
def _(show_source):
    from scripts.preprocess import make_text_tokenization_safe

    show_source(make_text_tokenization_safe)
    return (make_text_tokenization_safe,)


@app.cell
def _(chunked_data, make_text_tokenization_safe, preview_table):
    cleaned_data = []
    for _doc in chunked_data:
        _cleaned_doc = _doc.copy()
        _cleaned_doc["cleaned_content"] = make_text_tokenization_safe(_doc["content"])
        _cleaned_doc["metadata"]["cleaned_tokens"] = len(
            _cleaned_doc["cleaned_content"].split()
        )
        cleaned_data.append(_cleaned_doc)
    preview_table(cleaned_data, n=2)
    return (cleaned_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again we will store the cleaned data as a Weave Dataset named `chunked_data`.
    """)
    return


@app.cell
def _(cleaned_data, mo, weave, weave_client):
    mo.stop(weave_client is None)
    dataset = weave.Dataset(name="chunked_data", rows=cleaned_data)
    _ref = weave.publish(dataset)
    mo.md(f"Published Weave Dataset **chunked_data**: `{_ref.uri()}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![chunked_data](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/01_chunked_data.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vectorizing the data

    One of the key ingredient of most retrieval systems is to represent the given modality (text in our case) as a vector.

    This vector is a numerical representation representing the "content" of that modality (text).

    Text vectorization (text to vector) can be done using various techniques like [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model), [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) (Term Frequency-Inverse Document Frequency), and embeddings like [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [GloVe](https://nlp.stanford.edu/projects/glove/), and transformer based architectures like BERT and more, which capture the semantic meaning and relationships between words or sentences.

    In this chapter, we'll use TF-IDF (Term Frequency-Inverse Document Frequency) for vectorizing our contents. Here's why:

    - **Simplicity**: TF-IDF is straightforward to implement and understand, making it an excellent starting point for RAG systems.
    - **Efficiency**: It's computationally lightweight, allowing for quick processing of large document collections.
    - **No training required**: Unlike embedding models, TF-IDF doesn't need pre-training, making it easy to get started quickly.
    - **Interpretability**: The resulting vectors are directly related to word frequencies, making them easy to interpret.

    While more advanced methods like embeddings often provide better performance, especially for semantic understanding, we'll explore these in later chapters as we progress through the course.

    Let's download the `cleaned_data` artifact and use it to generate vectors for our chunks.
    """)
    return


@app.cell
def _(data_source, mo, preview_table, weave, weave_client):
    mo.stop(weave_client is None)
    _chunked_data_ref = (
        "weave:///rag-course/rag-course/object/chunked_data:Lt6M8qCUICD1JZTlMYzuDLTVtvFYESxvj3tcAIoPrtE"
        if data_source.value == "course"
        else "chunked_data:latest"
    )
    with mo.status.spinner(title="Fetching the `chunked_data` Dataset from Weave..."):
        chunked_data_ds = weave.ref(_chunked_data_ref).get()
    preview_table(chunked_data_ds.rows, n=2)
    return (chunked_data_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we will create a simple `Retriever` class. This class is responsible for vectorizing the chunks using the `index_data` method and provides a convenient method `search`, for querying the vector index using cosine distance similarity.

    - `index_data` will take a list of chunks and vectorize it using TF-IDF and store it as `index`.

    - `search` will take a `query` (question) and vectorize it using the same technique (TF-IDF in our case). It then computes the cosine distance between the query vector and the index (list of vectors) and pick the top `k` vectors from the index. These top `k` vectors represent the chunks that are closest (most relevant) to the `query`.

    ---

    Note that the `Retriever` class is inherited from `weave.Model`.

    A Model is a combination of data (which can include configuration, trained model weights, or other information) and code that defines how the model operates. By structuring your code to be compatible with this API, you benefit from a structured way to version your application so you can more systematically keep track of your experiments.

    To create a model in Weave, you need the following:

    - a class that inherits from weave.Model
    - type definitions on all attributes
    - a typed `predict`, `infer` or `forward` method with `@weave.op()` decorator.

    Imagine `weave.op()` to be a drop in replacement for `print` or logging statement.
    However, it does a lot more than just printing and logging by tracking the inputs and outputs of the function and storing them as Weave objects. In addition to state tracking you also get a nice weave UI to inspect the inputs, outputs, and other metadata.

    If you have not initialized a weave run by doing `weave.init`, the code will work as it is without any tracking.

    The `predict` method decorated with `weave.op()` will track the model settings along with the inputs and outputs anytime you call it.
    """)
    return


@app.cell
def _(show_source):
    from scripts.retriever import TFIDFRetriever

    show_source(TFIDFRetriever)
    return (TFIDFRetriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This simple TF-IDF based retriever serves as a good starting point, but for more complex applications, it can be extended or improved in several ways:

    - **Semantic search**: Implement embedding-based retrieval using dense vectors for better semantic understanding.
    - **Hybrid retrieval**: Combine TF-IDF with embedding-based methods to balance lexical and semantic matching.
    - **Query expansion**: Incorporate techniques like automatic query expansion to improve recall.
    - **Document ranking**: Implement more sophisticated ranking algorithms, such as BM25 and re-ranking models.
    - **Scalability**: For larger datasets, consider using approximate nearest neighbor search techniques or vector databases.

    As we progress through the course, we'll explore some of these concepts and gradually enhance our retriever to make it more robust and efficient for real-world applications.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's see our `TFIDFRetriever` in action. We will index our chunked data and then query the retriever to fetch related chunks from the index.
    """)
    return


@app.cell
def _(TFIDFRetriever, chunked_data_ds, mo):
    retriever = TFIDFRetriever()
    with mo.status.spinner(title="Indexing chunks with TF-IDF..."):
        retriever.index_data(list(map(dict, chunked_data_ds.rows)))
    return (retriever,)


@app.cell
def _(mo, retriever):
    query = "How do I use W&B to log metrics in my training script?"
    search_results = retriever.search(query)
    mo.vstack(
        [
            mo.md(f"**Query:** {query}"),
            mo.ui.table(search_results, selection=None, label="Top retrieved chunks"),
        ]
    )
    return query, search_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating a response

    There are two components of any RAG pipeline - a `Retriever` and a `ResponseGenerator`. Earlier, we designed a simple retriever. Here we are designing a simple `ResponseGenerator`.

    The `generate_response` method takes the user question along with the retrieved context (chunks) as inputs and makes a LLM call using the `model` and `prompt` (system prompt). This way the generated answer is grounded on the documentation (our usecase). In this course we are using Cohere's `command-r-08-2024` model.

    As earlier, we have wrapped this `ResponseGenerator` class with weave for tracking the inputs and the output.
    """)
    return


@app.cell
def _(show_source):
    from scripts.response_generator import SimpleResponseGenerator

    show_source(SimpleResponseGenerator)
    return (SimpleResponseGenerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is the system prompt. Consider this to be set of instructions on what to do with the user question and the retrieved contexts. In practice, the system prompt can be very detailed and involved (depending on the usecase) but we are showing a simple prompt. Later we will iterate on it and show how improving the system prompt improves the quality of the generated response.
    """)
    return


@app.cell
def _(NOTEBOOKS_DIR, mo):
    INITIAL_PROMPT = (NOTEBOOKS_DIR / "prompts" / "initial_system.txt").read_text()
    mo.md(f"```text\n{INITIAL_PROMPT}\n```")
    return (INITIAL_PROMPT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's generate the response for the question "How do I use Weights & Biases to log metrics in my training script?". We have already retrieved the context in the previous section and passing both the question and the context to the `generate_response` method.
    """)
    return


@app.cell
def _(mo):
    answer_button = mo.ui.run_button(label="Generate answer (calls Cohere)")
    answer_button
    return (answer_button,)


@app.cell
def _(
    INITIAL_PROMPT,
    SimpleResponseGenerator,
    answer_button,
    mo,
    query,
    search_results,
):
    mo.stop(
        not answer_button.value,
        mo.md("*Click the button above to generate the answer.*"),
    )
    _response_generator = SimpleResponseGenerator(
        model="command-r-08-2024", prompt=INITIAL_PROMPT
    )
    answer = _response_generator.generate_response(query, search_results)
    mo.md(answer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple Retrieval Augmented Generation (RAG) Pipeline

    Finally, we will bring everything together. As stated earlier, a RAG pipeline primarily consists of a retriever and a response generator.

    We define a class `SimpleRAGPipeline` which combines the steps of retrieval and response generation.

    We'll define a `predict` method that takes the user query, retrieves relevant context using the retriever and finally synthesizes a response using the response generator.

    We'll also define a few convinence methods to format the documents retrieved from the retriever and create a system prompt for the response generator.
    """)
    return


@app.cell
def _(show_source):
    from scripts.rag_pipeline import SimpleRAGPipeline

    show_source(SimpleRAGPipeline)
    return (SimpleRAGPipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us initialize the `RAGPipeline`.
    """)
    return


@app.cell
def _(INITIAL_PROMPT, SimpleRAGPipeline, SimpleResponseGenerator, retriever):
    # Initialize the response generator
    _response_generator = SimpleResponseGenerator(
        model="command-r-08-2024", prompt=INITIAL_PROMPT
    )
    # Bring them together as a RAG pipeline
    rag_pipeline = SimpleRAGPipeline(
        retriever=retriever, response_generator=_response_generator, top_k=5
    )
    return (rag_pipeline,)


@app.cell
def _(mo):
    rag_button = mo.ui.run_button(label="Run the RAG pipeline (calls Cohere)")
    rag_button
    return (rag_button,)


@app.cell
def _(mo, rag_button, rag_pipeline):
    mo.stop(
        not rag_button.value,
        mo.md("*Click the button above to run the full RAG pipeline.*"),
    )
    response = rag_pipeline.predict("How do I get get started with wandb?")
    mo.md(response)
    return


@app.cell(hide_code=True)
def _(mo, weave_client):
    mo.md(
        f"""
    Head over to [**your Weave traces**](https://wandb.ai/{weave_client.entity}/{weave_client.project}/weave/traces) — the same link printed with a 🍩 in the console output. This is the trace timeline for all the executions that happened in our simple RAG application. Go to the link and drill down to find everything that got tracked.

    ![weave trace timeline](https://raw.githubusercontent.com/wandb/edu/main/rag-advanced/images/01_weave_trace_timeline.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    ## Key Takeaways

    In this chapter, we've built a simple RAG pipeline from scratch. Here's what we've learned:

    - **Data Processing**: How to ingest, chunk, and clean data using W&B Artifacts and Weave
    - **Retrieval**: Implementing a basic TF-IDF based retriever
    - **Response Generation**: Using Cohere's API and `command-r-08-2024` model to generate responses based on retrieved context
    - **RAG Pipeline**: Combining retrieval and generation into a cohesive system
    - **Logging and Tracking**: Utilizing W&B Weave for efficient experiment tracking
    """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
