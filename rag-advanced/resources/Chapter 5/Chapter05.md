# Chapter 5

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter05.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-05} -->

## Retrieval and Re-ranking

In this chapter, we will focus on improving the retrieval quality and employ reranking to select the best context for our LLM.

In the last chapter, we added Query Enhancement module which was able to derive sub-queries from the user query. We retrieved contexts for these sub-queries. We did a basic deduplication of the total context to reduce the same context. However on a semantic level there might be few more contexts that we can drop thus reducing the input token cost.

When the same context is retrieved a few times, we can use this knowledge to fuse the confidence scores of such contexts and ask our LLM to focus more on them. Once can employ many strategies to improve the quality of contexts we provide our LLM.

Let's see a few here.


```
# @title Setup
!git clone https://github.com/wandb/edu.git
%cd edu/rag-advanced
!pip install -qqq -r requirements.txt
%cd notebooks

import nltk

nltk.download("wordnet")
```


```
import getpass
import os

os.environ["COHERE_API_KEY"] = getpass.getpass("Please enter your COHERE_API_KEY")
```


```
import asyncio

import nest_asyncio

nest_asyncio.apply()

import weave
```


```
WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)
```

We will download the chunked data from chapter 3


```
# Reload the data from Chapter 3
chunked_data = weave.ref(
    "weave:///rag-course/rag-course/object/chunked_data:Ij9KThmiZQ9ljpCm8rVXTJlCaAbY2qC0zX6UJkBWHQ0"
).get()
# uncomment the next line to get the chunked data from weave from your own project instead
# chunked_data = weave.ref("chunked_data:latest").get()

chunked_data.rows[:2]
chunked_data = list(map(dict, chunked_data.rows[:]))
```

## Embedding based retriever

In the past chapters we were using TF-IDF and BM25 based retrievers. The vector representation from these methods are not "dense" i.e, they were not trained on billions or trillions of tokens. LLMs today embed the tokens into a dense representation. These embeddings are learned during the training process and are used to represent the tokens in the high-dimensional vector space. They have a denser knowledge of different lingustic patterns.

One stratightforward way of improving our retriever is to use a `DenseRetriever` which uses the embedding model (Cohere embedding model here) to embed the chunks and uses the same embedding model to embed the query.

## Reranking the contexts

With more contexts it is important to pick the one that adds more knowledge about the given query. For this, a re-ranking model is used that calculates a matching score for a given query and document pair. This score can then be utilized to rearrange vector search results, ensuring that the most relevant results are prioritized at the top of the list. Cohere comes with it's own re-ranking model and is quite popular.



```
from scripts.reranker import CohereReranker
from scripts.retriever import DenseRetriever, DenseRetrieverWithReranker
from scripts.utils import display_source

display_source(DenseRetriever)
display_source(CohereReranker)
display_source(DenseRetrieverWithReranker)
```

Let's initialize the `DenseRetriever` and index the data.


```
dense_retriever = DenseRetriever()
dense_retriever.index_data(chunked_data)
```


```
from scripts.retrieval_metrics import IR_METRICS

eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

retrieval_evaluation = weave.Evaluation(
    name="Dense Retrieval Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 10},
)

dense_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(dense_retriever))
```

Let's initialize the `DenseRetrieverWithReranker` and index the data.


```
dense_retriever_rerank = DenseRetrieverWithReranker()
dense_retriever_rerank.index_data(chunked_data)
```


```
from scripts.retrieval_metrics import IR_METRICS

eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

retrieval_evaluation = weave.Evaluation(
    name="Dense Retrieval Rerank Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "top_k": 20, "top_n": 10},
)

dense_retrieval_scores = asyncio.run(
    retrieval_evaluation.evaluate(dense_retriever_rerank)
)
```

## Hybrid Retriever

Even though BM25 is an old model used for retrieval tasks, it is still the state-of-the-art on various benchmark. In machine learning, we ensemble a few weak classifiers to build a stronger classifier, we can adopt the same idea to our retriever pipeline.

Below we show the concept of hybrid retriever which uses two or more retrievers and retrievr chunks from all of them followed by re-ranking.

### Fusion reranking

Since the indiviual retrievers are gonna retrieve the same chunks most of the time, simple re-ranking will not be beneficial since the same chunk from each retriever will have similar score, thus the top_k after re-ranking will have more or less the same context.

Instead, we iterate through all the chunks and fuse the score of similar chunk. Finally we sort on the basis of this fused score and return top_k chunks.


```
from scripts.retriever import HybridRetrieverReranker

display_source(HybridRetrieverReranker)

hybrid_retriever = HybridRetrieverReranker()
```


```
hybrid_retriever.index_data(chunked_data)
```


```
eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

retrieval_evaluation = weave.Evaluation(
    name="Dense Retrieval Rerank Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "top_k": 20, "top_n": 10},
)

hybrid_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(hybrid_retriever))
```
