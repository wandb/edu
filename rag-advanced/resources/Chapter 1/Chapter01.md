# Chapter 1

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter01.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-01} -->

In this chapter, we will learn about building a simple RAG pipeline. We will mainly focus on how to preprocess and chunk the data followed by building a simple retrieval engine without using any fancy "Vector Index". The idea is to show the inner working of a retrieval pipeline and make you understand the workflow from a user query to a generated response using an LLM.

For this chapter you will need a Cohere API key and a W&B account. Run the next cell to pull the github repo and install the dependencies.


```
!git clone https://github.com/wandb/edu.git
%cd edu/rag-advanced
!pip install -qqq -r requirements.txt
%cd notebooks

import nltk

nltk.download("wordnet")
```

Now we are ready to start the chapter.

We will start by logging into W&B and setup the API keys in the environment. You should be able to get your Cohere API key from [here](https://dashboard.cohere.com/api-keys).


```
import getpass
import os

import wandb

os.environ["COHERE_API_KEY"] = getpass.getpass("Please enter your COHERE_API_KEY")
wandb.login()
```


```
import pathlib
from typing import List

import weave

from scripts.utils import display_source
```

Here, we will start a Weights and Biases (W&B) run. We will be using this to download a [W&B Artifact](https://docs.wandb.ai/guides/artifacts) called `wandb_docs`. This is the raw W&B documentation. W&B Artifacts is suited for versiong different data sources which needs preprocessing/cleaning.


```
WANDB_PROJECT = "rag-course"

run = wandb.init(
    project=WANDB_PROJECT,
    group="Chapter 1",
)
```

## Data ingestion

### Loading the data

Use [W&B Artifacts](https://docs.wandb.ai/guides/artifacts) to track and version data as the inputs and outputs of your W&B Runs. For example, a model training run might take in a dataset as input and produce a trained model as output. W&B Artifact is a powerful object storage with rich UI functionalities.

Below we are downloading an artifact named `wandb_docs` which will download 400 odd markdown files in your `../data/wandb_docs` directory. This is will be our data source.


```
documents_artifact = run.use_artifact(
    f"rag-course/dev/wandb_docs:latest", type="dataset"
)
data_dir = "../data/wandb_docs"

docs_dir = documents_artifact.download(data_dir)
```

Let's inspect the `../data/wandb_docs` directory and look at the name first 5 files. We should see that they are all in markdown (`.md`) file format.


```
docs_dir = pathlib.Path(docs_dir)
docs_files = sorted(docs_dir.rglob("*.md"))

print(f"Number of files: {len(docs_files)}\n")
print("First 5 files:\n{files}".format(files="\n".join(map(str, docs_files[:5]))))
```

Lets look at an example file. We can take the first element of the list (`docs_files`) and use the `Path.read_text` method to get the decoded contents of the file as a string.


```
print(docs_files[0].read_text())
```

💡 Looking at the example, we see some structure and format to it.

It is always a good practice to look through few examples to see if there is any pattern to your data source.
It helps to come up with better preprocessing steps and chunking strategies.

Now, let's store our documents as dictionaries with content (raw text) and metadata.

Metadata is extra information for that data point which can be used to group together similar data points, or filter out a few data points.
We will see in future chapters the importance of metadata and why it should not be ignored while building the ingestion pipeline.

The metadata can be derived (`raw_tokens`) or is inherent (`source`) to the data point.

Note that we are simply doing word counting and calling it `raw_tokens`.
In practice we would be using a [tokenizer](https://docs.cohere.com/docs/tokens-and-tokenizers) to calculate the token counts but this naive calculation is an okay approximation for now.


```
# We'll store the files as dictionaries with some content and metadata
data = []
for file in docs_files:
    content = file.read_text()
    data.append(
        {
            "content": content,
            "metadata": {
                "source": str(file.relative_to(docs_dir)),
                "raw_tokens": len(content.split()),
            },
        }
    )
data[:2]
```

Checking the total number of tokens of your data source is a good practice. In this case, the total tokens is more than 200k. Surely, most LLM providers cannot process these many tokens. Building a RAG is justified in such cases.


```
total_tokens = sum(map(lambda x: x["metadata"]["raw_tokens"], data))
print(f"Total Tokens in dataset: {total_tokens}")
```

## W&B Weave

In the previous section we used a W&B Artifact to download the source documents.

We could have used it to log our processed data but instead we will use W&B Weave for the task.

Why?

- W&B Weave is standalone and doesn't need backward compatibility with core W&B offerings.
- W&B Weave is designed for modern LLMOps use case.
- We would like to keep the data, "models" (function, API, collection of parser, extrators, and more) and evaluators in "one single source of truth" managed by W&B Weave.

What?

W&B Weave is a lightweight toolkit for tracking and evaluating LLM applications:

- Log and debug language model inputs, outputs, and traces
- Build rigorous, apples-to-apples evaluations for language model use cases
- Organize all the information generated across the LLM workflow, from experimentation to evaluations to production


The newly created list of dictionaries with `content` and `metadata` will now be logged as a W&B Weave Dataset called `raw_data`. Notice that out processed data is a list of dicts.

Let's initialize W&B Weave. Once intialized, we will start tracking (more on it later) the inputs and the outputs along with underlying attributes (model name, top_k, etc.).


```
WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)
```


```
# build weave dataset
raw_data = weave.Dataset(name="raw_data", rows=data)

# publish the dataset
weave.publish(raw_data)
```

![raw_data](../images/01_raw_data.png)

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


```
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
```

We will use the `raw_data` artifact we just published as W&B Weave Dataset as the input to the chunking function.


```
# download the `raw_data` Dataset
raw_data = weave.ref(
    "weave:///rag-course/rag-course/object/raw_data:nuZosGsP58MXKxhluN2hzvKK9XB8xSxlTuOBmEzWzLo"
).get()
# uncomment the next line to get the raw data from weave from your own project instead
# raw_data = weave.ref("raw_data:v0").get()

# this is how we index into the data
print(raw_data.rows[:2])
```

A Weave Dataset is compatible with the rest of the Weave workflow and is pythonic.

Let us chunk each document in the raw data Artifact. We create a new list of dictionaries with the chuked text (`content`) and with `metadata`.


```
chunked_data = []
for doc in raw_data.rows:
    chunks = split_into_chunks(doc["content"])
    for chunk in chunks:
        chunked_data.append(
            {
                "content": chunk,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "raw_tokens": len(chunk.split()),
                },
            }
        )

print(chunked_data[:2])
```

### Cleaning the data

Cleaning up data is crucial for most ML pipelines and applies to a RAG/Agentic pipeline as well.
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


```
from scripts.preprocess import make_text_tokenization_safe

display_source(make_text_tokenization_safe)
```


```
cleaned_data = []
for doc in chunked_data:
    cleaned_doc = doc.copy()
    cleaned_doc["cleaned_content"] = make_text_tokenization_safe(doc["content"])
    cleaned_doc["metadata"]["cleaned_tokens"] = len(
        cleaned_doc["cleaned_content"].split()
    )
    cleaned_data.append(cleaned_doc)
print(cleaned_data[:2])
```

Again we will store the cleaned data as a Weave Dataset named `chunked_data`.


```
dataset = weave.Dataset(name="chunked_data", rows=cleaned_data)
weave.publish(dataset)
```

![chunked_data](../images/01_chunked_data.png)

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


```
chunked_data = weave.ref(
    "weave:///rag-course/rag-course/object/chunked_data:Lt6M8qCUICD1JZTlMYzuDLTVtvFYESxvj3tcAIoPrtE"
).get()
# uncomment the next line to get the chunked data from weave from your own project instead
# chunked_data = weave.ref("chunked_data:v0").get()
print(chunked_data.rows[:2])
```

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


```
from scripts.retriever import TFIDFRetriever

display_source(TFIDFRetriever)
```

This simple TF-IDF based retriever serves as a good starting point, but for more complex applications, it can be extended or improved in several ways:

- **Semantic search**: Implement embedding-based retrieval using dense vectors for better semantic understanding.
- **Hybrid retrieval**: Combine TF-IDF with embedding-based methods to balance lexical and semantic matching.
- **Query expansion**: Incorporate techniques like automatic query expansion to improve recall.
- **Document ranking**: Implement more sophisticated ranking algorithms, such as BM25 and re-ranking models.
- **Scalability**: For larger datasets, consider using approximate nearest neighbor search techniques or vector databases.

As we progress through the course, we'll explore some of these concepts and gradually enhance our retriever to make it more robust and efficient for real-world applications.

Now, let's see our `TFIDFRetriever` in action. We will index our chunked data and then query the retriever to fetch related chunks from the index.


```
retriever = TFIDFRetriever()
retriever.index_data(list(map(dict, chunked_data.rows)))
```


```
query = "How do I use W&B to log metrics in my training script?"
search_results = retriever.search(query)
for result in search_results:
    print(result)
```

## Generating a response

There are two components of any RAG pipeline - a `Retriever` and a `ResponseGenerator`. Earlier, we designed a simple retriever. Here we are designing a simple `ResponseGenerator`.

The `generate_response` method takes the user question along with the retrieved context (chunks) as inputs and makes a LLM call using the `model` and `prompt` (system prompt). This way the generated answer is grounded on the documentation (our usecase). In this course we are using Cohere's `command-r` model.

As earlier, we have wrapped this `ResponseGenerator` class with weave for tracking the inputs and the output.


```
from scripts.response_generator import SimpleResponseGenerator

display_source(SimpleResponseGenerator)
```

Below is the system prompt. Consider this to be set of instructions on what to do with the user question and the retrieved contexts. In practice, the system prompt can be very detailed and involved (depending on the usecase) but we are showing a simple prompt. Later we will iterate on it and show how improving the system prompt improves the quality of the generated response.


```
INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
print(INITIAL_PROMPT)
```

Let's generate the response for the question "How do I use W&B to log metrics in my training script?". We have already retrieved the context in the previous section and passing both the question and the context to the `generate_response` method.


```
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)
answer = response_generator.generate_response(query, search_results)
print(answer)
```

## Simple Retrieval Augmented Generation (RAG) Pipeline

Finally, we will bring everything together. As stated earlier, a RAG pipeline primarily consists of a retriever and a response generator.

We define a class `SimpleRAGPipeline` which combines the steps of retrieval and response generation.

We'll define a `predict` method that takes the user query, retrieves relevant context using the retriever and finally synthesizes a response using the response generator.

We'll also define a few convinence methods to format the documents retrieved from the retriever and create a system prompt for the response generator.


```
from scripts.rag_pipeline import SimpleRAGPipeline

display_source(SimpleRAGPipeline)
```

Let us initialize the `RAGPipeline`.


```
# Initialize the response generator
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)

# Bring them together as a RAG pipeline
rag_pipeline = SimpleRAGPipeline(
    retriever=retriever, response_generator=response_generator, top_k=5
)
```


```
response = rag_pipeline.predict("How do I get get started with wandb?")
print(response, sep="\n")
```

Click on the link starting with a 🍩. This is the trace timeline for all the executions that happened in our simple RAG application. Go to the link and drill down to find everything that got tracked.

![weave trace timeline](../images/01_weave_trace_timeline.png)

## Key Takeaways

In this chapter, we've built a simple RAG pipeline from scratch. Here's what we've learned:

- **Data Processing**: How to ingest, chunk, and clean data using W&B Artifacts and Weave
- **Retrieval**: Implementing a basic TF-IDF based retriever
- **Response Generation**: Using Cohere's API and `command-r` model to generate responses based on retrieved context
- **RAG Pipeline**: Combining retrieval and generation into a cohesive system
- **Logging and Tracking**: Utilizing W&B Weave for efficient experiment tracking
