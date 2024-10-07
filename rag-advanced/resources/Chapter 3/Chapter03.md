# Chapter 3 

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter03.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-03} -->

## Data Ingestion and Preprocessing

At its core, RAG (Retrieval-Augmented Generation) is a method for integrating private data with pre-trained, instruction-tuned language models. The effectiveness of your RAG system is directly proportional to the quality of your data ingestion pipeline.

Data ingestion encompasses both data sources and preprocessing. As with most machine learning systems, the principle of "garbage in, garbage out" applies to LLMs. Therefore, optimizing your data ingestion pipeline is crucial for RAG efficacy.

Key considerations for efficient data ingestion:
1. Periodic updates: Implement a system that can seamlessly update when data sources change.
2. Quality control: Ensure data cleanliness and relevance.
3. Scalability: Design the pipeline to handle increasing data volumes.


**Tip**: When building your data ingestion pipeline, start with a small, representative sample of your data. This allows you to quickly iterate on your preprocessing steps and catch potential issues early. Focus on creating an end-to-end working system before optimizing specific components like chunk size, parsing strategies, or data formats (e.g., markdown, HTML, plain text).


To begin, execute the following cell to clone the repository and install dependencies:


```python
!git clone https://github.com/wandb/edu.git
%cd edu/rag-advanced
!pip install -qqq -r requirements.txt
%cd notebooks

import nltk

nltk.download("wordnet")
nltk.download("punkt")
```

With the setup complete, we can now proceed with the chapter content.

Initial steps:
1. Log in to Weights & Biases (W&B)
2. Configure environment variables for API access

To obtain your Cohere API key, visit the [Cohere API dashboard](https://dashboard.cohere.com/api-keys).


```python
import getpass
import os

os.environ["COHERE_API_KEY"] = getpass.getpass("Please enter your COHERE_API_KEY")
```


```python
import asyncio

import nest_asyncio

nest_asyncio.apply()
import numpy as np
import weave

from scripts.utils import display_source
```


```python
WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)
```

**Best Practice**: Document your data preprocessing steps meticulously. This not only helps with reproducibility but also makes it easier to debug issues and optimize your pipeline later.

## Data Preparation

We'll begin our journey with raw data processing. 

Our first step is to retrieve the most recent `raw_data` we logged into weave. We had logged this in Chapter 1 and we will use the same dataset as our starting point.


```python
# download the `raw_data` Dataset from chapter 1
raw_data = weave.ref(
    "weave:///rag-course/rag-course/object/raw_data:nuZosGsP58MXKxhluN2hzvKK9XB8xSxlTuOBmEzWzLo"
).get()
# uncomment the next line to get the raw data from weave from your own project instead
# raw_data = weave.ref("raw_data:v0").get()

# this is how we index into the data
print(raw_data.rows[:2])
raw_data = list(map(dict, raw_data.rows[:]))
```

In chapter 1, we naively counted each word (as they appear in English text) as one token (`raw_tokens`). Now let's update to using the correct token counting strategy (`tokens`).


We will use [Cohere's tokenizer](https://docs.cohere.com/docs/tokens-and-tokenizers) to calculate the accurate number of tokens per document in our `raw_data`. Both the correct token count and word count will be stored as metadata for each document.


In RAG systems, accurate tokenization is crucial for proper text processing and context management. Let's examine two key functions: `tokenize_text` and `length_function`.

**The `tokenize_text` Function**: This function tokenizes input text using Cohere's tokenization API. Here's how it works:

1. It initializes a Cohere client using an API key stored in environment variables.
2. It calls the `tokenize` method of the Cohere client, passing:
   - The input `text`
   - The specified `model` (defaulting to "command-r")
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


```python
from scripts.utils import (TOKENIZERS, get_special_tokens_set, length_function,
                           tokenize_text)

# this is the function that will tokenize the text
display_source(tokenize_text)
# this is the function that calculates the number of tokens
display_source(length_function)
```


```python
for doc in raw_data[:]:
    doc["metadata"]["words"] = doc["metadata"].pop("raw_tokens")
    doc["metadata"]["tokens"] = length_function(doc["content"])
raw_data[:2]
```

As you can see above, the `words` count (used in Chapter 1) is quite different from the actual `tokens` count. This discrepancy highlights the importance of accurate token counting in RAG systems. Knowing the correct number of tokens is crucial for several reasons:

1. It helps decide whether to build a RAG pipeline or ingest the whole document into an LLM, especially now that many top LLMs support long context windows.
2. It informs the optimal chunk size for efficient processing.
3. It ensures efficient use of context windows and helps manage costs associated with token-based API calls.
4. It aids in predicting and controlling response generation length, which is vital for maintaining coherent and relevant outputs.

By accurately counting tokens, we can make informed decisions about our RAG system's architecture and optimize its performance and cost-effectiveness.

**Tip**: Different tokenizers may produce slightly different results. Always use the same tokenizer that your target LLM uses to ensure consistency between preprocessing and model input.

## Pre-processing

Now that we have our raw data prepared and correctly tokenized, the next crucial step is to pre-process this data. Pre-processing is essential for removing extraneous information and formatting that could interfere with our language model's understanding of the content.

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


```python
from scripts.preprocess import (convert_contents_to_text,
                                make_text_tokenization_safe)

# this is the function that converts the markdown to html
display_source(convert_contents_to_text)
# this is the function that cleans the text
display_source(make_text_tokenization_safe)
```

We are converting the raw markdown documents to text and making them tokenization-safe. This process involves removing special tokens that could interfere with the tokenization process. Let's examine the first 5 special tokens to understand what's being removed.

After processing, you'll notice that the `parsed_tokens` count is smaller compared to the original `tokens` count. This reduction is expected and indicates that we've successfully removed extraneous formatting and special characters, resulting in a cleaner text representation that's more suitable for our LLM pipeline.

This step is crucial for ensuring that our input data is optimized for tokenization and subsequent processing by the language model.


```python
special_tokens_set = get_special_tokens_set(TOKENIZERS["command-r"])
print(list(special_tokens_set)[:5])

parsed_data = []

for doc in raw_data:
    parsed_doc = doc.copy()
    content = convert_contents_to_text(doc["content"])
    parsed_doc["parsed_content"] = make_text_tokenization_safe(
        content, special_tokens_set=special_tokens_set
    )
    parsed_doc["metadata"]["parsed_tokens"] = length_function(
        parsed_doc["parsed_content"]
    )
    parsed_data.append(parsed_doc)
parsed_data[:2]
```

Again, we can store the parsed data as a weave Dataset


```python
# build weave dataset
parsed_data = weave.Dataset(name="parsed_data", rows=parsed_data)

# publish the dataset
weave.publish(parsed_data)
```

## Data Chunking

With our data cleaned and pre-processed, we're ready to move on to the next critical step: chunking. Chunking involves breaking down our processed documents into smaller, manageable pieces. This step is crucial for several reasons:

1. It allows us to retrieve more relevant information
2. It helps manage token limits in language models
3. It can improve the overall efficiency of our RAG system

In this section, we'll explore different chunking strategies and implement a semantic chunking approach, which aims to preserve the context and meaning of our text while splitting it into appropriate segments.

We can split the processed data into smaller chunks. This approach serves two purposes:
1. Reduce input token cost by sending only the required data for generation.
2. Limit context to ensure the LLM focuses on relevant details.

While sending the entire document to the LLM is possible, it depends on the total token count and the nature of your use case. This approach can be costlier but is a good starting point.

### Semantic Chunking

Various chunking strategies exist, such as splitting after n words/tokens or on headers. It's advisable to experiment with these simple strategies before moving to more sophisticated ones.

Below, we implement semantic chunking, a sophisticated strategy that has proven effective in practice. This method groups similar sentences into chunks:

1. Split the text into sentences using the [BlingFire](https://github.com/microsoft/BlingFire) library.
2. Group and combine chunks based on semantic similarity.


Semantic chunking offers key advantages over simpler methods like fixed-length splitting:

1. **Improved Relevance**: Groups related sentences, increasing the likelihood of retrieving complete, relevant information.
2. **Context Preservation**: Maintains logical flow within chunks, crucial for accurate LLM understanding and generation.
3. **Adaptive Segmentation**: Creates variable-length chunks that better represent the text's natural structure and content organization.

By keeping related information together, semantic chunking optimizes retrieval accuracy and enhances the RAG system's ability to provide contextually appropriate responses.

For more information on the chunking strategy used, refer to this [research article on evaluating chunking](https://research.trychroma.com/evaluating-chunking).


```python
# download the `parsed_data` Dataset
parsed_data = weave.ref(
    "weave:///rag-course/rag-course/object/parsed_data:UhWHAwXzvIcYaZ3X1x4eX2KDyYhCM4TPSsj8Oq8dLq4"
).get()
# uncomment the next line to get the parsed data from weave from your own project instead
# parsed_data = weave.ref("parsed_data:v0").get()

# this is how we index into the data
print(parsed_data.rows[:2])

parsed_data = list(map(dict, parsed_data.rows[:]))
parsed_data[:2]
```


```python
from scripts.chunking import chunk_documents

# this is the function that chunks the documents
display_source(chunk_documents)
```

Since we are doing semantic chunking, the chunking process can take a while. For now, let's just take the first 5 documents and chunk them.


```python
sample_chunked_data = chunk_documents(parsed_data[:5])
sample_chunked_data[:2]
```

For the rest of the data, we'll retrieve the chunked data from weave.


```python
# fetch the chunked data
chunked_data = weave.ref(
    "weave:///rag-course/rag-course/object/chunked_data:Ij9KThmiZQ9ljpCm8rVXTJlCaAbY2qC0zX6UJkBWHQ0"
).get()
# uncomment the next line to get the chunked data from weave from your own project instead
# chunked_data = weave.ref("chunked_data:latest").get()

# this is how we index into the data
print(chunked_data.rows[:2])

chunked_data = list(map(dict, chunked_data.rows[:]))
```


```python
mean_chunk_size = np.mean([doc["metadata"]["parsed_tokens"] for doc in chunked_data])
std_chunk_size = np.std([doc["metadata"]["parsed_tokens"] for doc in chunked_data])
print(f"Mean chunk size: {mean_chunk_size}, Std chunk size: {std_chunk_size}")
```


```python
# if you run your own chunking method, you can publish the chunked data in a weave Dataset
# # Again, we'll store the chunked data in a weave Dataset
# chunked_data = weave.Dataset(name="chunked_data", rows=chunked_data)

# # publish the dataset
# weave.publish(chunked_data)
```

**Tip**: Experiment with different chunking strategies and sizes. The optimal approach often depends on your specific use case and the nature of your documents. Monitor how changes in chunking affect both retrieval accuracy and LLM performance.

## Exploring Alternative Retrieval Methods

Now that we have our data prepared, pre-processed, and chunked, it's time to focus on how we retrieve this information. While we've previously used TF-IDF as our baseline retrieval method, it's important to explore alternatives that might offer improved performance.

In this section, we'll introduce the BM25 (Best Matching 25) retriever, a more sophisticated approach to information retrieval. BM25 is an evolution of TF-IDF that addresses some of its limitations. While TF-IDF simply weighs term frequency against document frequency, BM25 incorporates document length normalization and term frequency saturation. This means BM25 can better handle varying document lengths and prevents common terms from dominating the relevance score. As a result, BM25 often provides more nuanced rankings, especially for longer documents or queries with multiple terms.

By comparing the performance of BM25 against our existing TF-IDF retriever, we can gain valuable insights into:

1. The strengths and weaknesses of different retrieval algorithms in our specific use case
2. The impact of more sophisticated ranking functions on RAG performance
3. Potential areas for further optimization in our retrieval pipeline

In our RAG pipeline, this could lead to more contextually relevant retrievals, potentially improving the quality of the generated responses. Let's implement the BM25 retriever and set up a comparative analysis with our TF-IDF baseline:


```python
from scripts.rag_pipeline import SimpleRAGPipeline
from scripts.response_generator import SimpleResponseGenerator
from scripts.retriever import BM25Retriever, TFIDFRetriever

display_source(BM25Retriever)
```


```python
bm25_retriever = BM25Retriever()
bm25_retriever.index_data(chunked_data)

tfidf_retriever = TFIDFRetriever()
tfidf_retriever.index_data(chunked_data)
```

The rest of the RAG pipeline remains unchanged. We'll use the same response generator and overall structure as before, allowing us to isolate the impact of our new retrieval method. This approach ensures a fair comparison between the TF-IDF and BM25 retrievers within our existing framework.


```python
INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)
bm25_rag_pipeline = SimpleRAGPipeline(
    retriever=bm25_retriever, response_generator=response_generator, top_k=5
)
tfidf_rag_pipeline = SimpleRAGPipeline(
    retriever=tfidf_retriever, response_generator=response_generator, top_k=5
)
```

## Evaluate and compare the changes

With our new retrieval method implemented, it's crucial to evaluate its performance and compare it to our baseline. This evaluation will help us understand the impact of our changes and determine whether the BM25 retriever offers improvements over the TF-IDF method.

In this section, we'll use the evaluation dataset and metrics from Chapter 2 to assess both the retrieval performance and the overall RAG pipeline performance with each retriever. This comprehensive evaluation will provide valuable insights into the effectiveness of our improvements.


We are primarily interested in two aspects:
1. The impact of pre-processing on retrieval metrics
2. The effect of different retrieval methods on response metrics

To address these points, we will evaluate:
1. Retrieval metrics for both TF-IDF and BM25 retrievers
2. Response metrics for the RAG pipeline using both retrievers

This comprehensive evaluation will provide insights into the performance of individual retrieval methods and the overall RAG pipeline.

We'll begin by fetching the evaluation dataset and metrics used in Chapter 2. Using Weave, we'll retrieve the dataset and metrics, then run the evaluation using [Weave Evaluations](https://weave-docs.wandb.ai/guides/core-types/evaluations/).


```python
from scripts.response_metrics import ALL_METRICS as RESPONSE_METRICS
from scripts.retrieval_metrics import ALL_METRICS as RETRIEVAL_METRICS
```


```python
eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

print("Number of evaluation samples: ", len(eval_dataset.rows))
```


```python
retrieval_evaluation = weave.Evaluation(
    name="Retrieval_Evaluation",
    dataset=eval_dataset,
    scorers=RETRIEVAL_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
)
bm25_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(bm25_retriever))
tfidf_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(tfidf_retriever))
```

![compare_retrievers](../images/03_compare_retrievers.png)

BM25 outperforms TFIDF in most relevance metrics:
- Higher MAP, NDCG, and MRR indicate better ranking and relevance.
- Improved precision and recall suggest a better balance in retrieving relevant documents.
- Slightly better hit rate.

TFIDF advantages:
- Significantly lower latency, beneficial for time-sensitive applications.

Trade-offs:
- BM25 offers superior retrieval performance at the cost of higher latency.
- TFIDF is faster but generally less accurate in retrieval.


```python
response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset,
    scorers=RESPONSE_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)
bm25_response_scores = asyncio.run(response_evaluations.evaluate(bm25_rag_pipeline))
tfidf_response_scores = asyncio.run(response_evaluations.evaluate(tfidf_rag_pipeline))
```

![compare_retriever_responses](../images/03_compare_retriever_responses.png)

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

# Key Takeaways

1. Data Quality is Crucial: The effectiveness of a RAG system heavily depends on the quality of data ingestion and preprocessing. "Garbage in, garbage out" applies to LLMs as well.

2. Accurate Tokenization: Using the correct tokenization strategy is essential for proper text processing and context management. The actual token count often differs significantly from simple word counts.

3. Preprocessing Importance: Cleaning raw data by removing markdown elements, special characters, and extra whitespace is crucial for optimizing LLM input and improving retrieval accuracy.

4. Semantic Chunking: This advanced chunking strategy groups similar sentences, preserving context and improving retrieval relevance compared to simpler methods like fixed-length splitting.

5. Retrieval Method Comparison: Exploring alternative retrieval methods, such as BM25 vs. TF-IDF, can lead to significant improvements in RAG system performance.

6. Evaluation is Key: Regularly evaluating both retrieval and response metrics is crucial for understanding the impact of changes and optimizing the RAG pipeline.

7. Experimentation: The optimal approach for data preprocessing, chunking, and retrieval often depends on the specific use case. Experimentation is encouraged to find the best configuration for your RAG system.

8. Scalability and Updates: When building a data ingestion pipeline, consider scalability and the ability to handle periodic updates as data sources change.

# Exercise

1. Add more data sources to the RAG system. - Add Jupyter Notbooks from the See wandb/examples repo.
2. Use a different chunking method. - Try your own parsing and chunking method.
3. Use a small-to-big retrieval method. Where we embed small documents but retrieve big documents -> You can add the parent document to the metadata and modify the `Retriever.search` method.
