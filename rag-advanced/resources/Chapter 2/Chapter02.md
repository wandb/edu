# Chapter 2:

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter02.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-02} -->

**Comprehensive Evaluation Strategies**

In this chapter, we will evaluate the two main components of a RAG pipeline - retriever and response generator.

Evaluating the retriever can be considered component evaluation. Depending on your RAG pipeline, there can be a few components and for ensuring robustness of your system, it is recommended to come up with evaluation for each components.

We start off by setting up the required packages.



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

from scripts.utils import display_source
```

In this chapter we will also use W&B Weave for our evaluation purposes. The `weave.Evaluation` class is a light weight class that can be used to evaluate the performance of a `weave.Model` on a `weave.Dataset`. We will go into more details.

We first initialize a weave client which can track both the traces and the evaluation scores.


```
WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)
```

## Collecting data for evaluation

We are using a subset of the evaluation dataset we had created for wandbot.

Learn more about how we created the evaluation dataset here:

- [How to Evaluate an LLM, Part 1: Building an Evaluation Dataset for our LLM System](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-Evaluate-an-LLM-Part-1-Building-an-Evaluation-Dataset-for-our-LLM-System--Vmlldzo1NTAwNTcy)
- [How to Evaluate an LLM, Part 2: Manual Evaluation of Wandbot, our LLM-Powered Docs Assistant](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-Evaluate-an-LLM-Part-2-Manual-Evaluation-of-Wandbot-our-LLM-Powered-Docs-Assistant--Vmlldzo1NzU4NTM3)

The main take away from these reports are:

- we first deployed wandbot for internal usage based on rigorous eyeballing based evalution.
- the user query distribution was throughly analyized and clustered. we samples a good representative queries from these clusters and created a gold standard set of queries.
- we then used in-house MLEs to perform manual evaluation using Argilla. Creating such evaluation platforms are easy.
- To summarize, speed is the key here. Use whatever means you have to create a meaningful eval set.

The evaluation samples are logged as [`weave.Dataset`](https://wandb.github.io/weave/guides/core-types/datasets/). `weave.Dataset` enable you to collect examples for evaluation and automatically track versions for accurate comparisons.

Below we will download the latest version locally with a simple API.


```
# Easy eval dataset with 20 samples.
eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

print("Number of evaluation samples: ", len(eval_dataset.rows))
```

Iterating through each sample is easy.

We have the question, ground truth answer and ground truth contexts.


```
dict(eval_dataset.rows[0])
```

## Evaluating the Retriever

The fundamental idea of evaluating a retriever is to check how well the retrieved content matches the expected contents. For evaluating a RAG pipeline end to end, we need query and ground truth answer pairs. The ground truth answer must be grounded on some "ground" truth chunks. This is a search problem, it's easiest to start with tradiaional Information retrieval metrics.

You might already have access to such evaluation dataset depending on the nature of your application or you can synthetically build one. To build one you can retrieve random documents/chunks and ask an LLM to generate query-answer pairs - the underlying documents/chunks will act as your ground truth chunk.

In the sections below, we will look at different metrics that can be used to evaluate the retriever we built in the last chapter.

First let us download the chunked data (from chapter 1) and index it using a simple TFIDF based retriever.



```
# Reload the data from Chapter 1
chunked_data = weave.ref(
    "weave:///rag-course/rag-course/object/chunked_data:Lt6M8qCUICD1JZTlMYzuDLTVtvFYESxvj3tcAIoPrtE"
).get()
# uncomment the next line to get the chunked data from weave from your own project instead
# chunked_data = weave.ref("chunked_data:v0").get()
print("Number of chunked data: ", len(chunked_data.rows))
chunked_data.rows[:2]
```

We will import the `TFIDFRetriever` which is an instance of `weave.Model` and index the chunked data from the last chapter.


```
from scripts.retriever import TFIDFRetriever

display_source(TFIDFRetriever)

retriever = TFIDFRetriever()
retriever.index_data(list(map(dict, chunked_data.rows)))
```

## Metrics to evaluate retriever

We can evaluate a retriever using traditional ML metrics. We can also evaluate by using a powerful LLM (next section).

Below we are importing both traditional metrics and LLM as a judge metric from the `scripts/retrieval_metrics.py` file.


```
from scripts.retrieval_metrics import IR_METRICS
from scripts.utils import display_source
```

Let us first understand the basic traditional metrics we will be using. Each metric expects a `model_output` which is a list of retrieved chunks from the retriever and `contexts` which is a list of ground truth contexts.


```
for scorer in IR_METRICS:
    display_source(scorer)
```

#### Evaluating retrieval on other metrics


```
retrieval_evaluation = weave.Evaluation(
    name="Retrieval_Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
)

retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(retriever))
```

### Using an LLM evaluator for evaluating retriever

**ref: https://arxiv.org/pdf/2406.06519**

How do we evaluate if we don't have any ground truth?

We can use a powerful LLM as a judge to evaluate the retriever.



```
from scripts.retrieval_metrics import LLM_METRICS

for metric in LLM_METRICS:
    display_source(metric)
```


```
retrieval_evaluation = weave.Evaluation(
    name="LLM_Judge_Retrieval_Evaluation",
    dataset=eval_dataset,
    scorers=LLM_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
)
retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(retriever))
```

## Evaluating the Response


```
from scripts.rag_pipeline import SimpleRAGPipeline
from scripts.response_generator import SimpleResponseGenerator

INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)
rag_pipeline = SimpleRAGPipeline(
    retriever=retriever, response_generator=response_generator, top_k=5
)
```


```
from scripts.response_metrics import NLP_METRICS

for scorer in NLP_METRICS:
    display_source(scorer)
```


```
response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset,
    scorers=NLP_METRICS[:-1],
    preprocess_model_input=lambda x: {"query": x["question"]},
)

response_scores = asyncio.run(response_evaluations.evaluate(rag_pipeline))
```

### Using an LLM as a Response Judge

Some metrics cannot be defined objectively and are particularly useful for more subjective or complex criteria.
We care about correctness, faithfulness, and relevance.

- **Answer Correctness** - Is the generated answer correct compared to the reference and thoroughly answers the user's query?
- **Answer Relevancy** - Is the generated answer relevant and comprehensive?
- **Answer Factfulness** - Is the generated answer factually consistent with the context document?



```
from scripts.response_metrics import LLM_METRICS

for metric in LLM_METRICS:
    display_source(metric)
```


```
correctness_evaluations = weave.Evaluation(
    name="Correctness_Evaluation",
    dataset=eval_dataset,
    scorers=LLM_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)

response_scores = asyncio.run(correctness_evaluations.evaluate(rag_pipeline))
```

## Exercise

1. Implement the `Relevance` and `Faithfulness` evaluators and evaluate the pipeline on all the dimensions.
2. Generate and share a W&B report with the following sections in the form of tables and charts:
    
    - Summary of the evaluation
    - Retreival Evaluations
        - IR Metrics
        - LLM As a Retrieval Judge Metric
    - Response Evalations
        - Traditional NLP Metrics
        - LLM Judgement Metrics
    - Overall Evalations
    - Conclusion

