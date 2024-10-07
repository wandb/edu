# Chapter 4

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter04.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-04} -->

## Query Enhancement

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

This hands-on experience will deepen your understanding of advanced RAG concepts and prepare you to implement these techniques in your own projects.

Let's begin by setting up our environment and importing the necessary libraries.

To begin, execute the following cell to clone the repository and install dependencies:


```python
!git clone https://github.com/wandb/edu.git
%cd edu/rag-advanced
!pip install -qqq -r requirements.txt
%cd notebooks

import nltk

nltk.download("wordnet")
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

import cohere
import weave
from IPython.display import Markdown
```


```python
WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)
```

## Data Loading

We'll start by loading the semantically chunked data from Chapter 3. As a reminder, semantic chunking is an technique that groups related sentences together, preserving context and improving retrieval relevance.

This chunked data will serve as the input for the knowledge base for our RAG pipeline, allowing us to compare the effectiveness of our query enhancement techniques against a baseline system.

Let's load the data and take a look at the first few chunks:


```python
# Reload the data from Chapter 3
chunked_data = weave.ref(
    "weave:///rag-course/rag-course/object/chunked_data:Ij9KThmiZQ9ljpCm8rVXTJlCaAbY2qC0zX6UJkBWHQ0"
).get()
# uncomment the next line to get the chunked data from weave from your own project instead
# chunked_data = weave.ref("chunked_data:latest").get()

print(chunked_data.rows[:2])

chunked_data = list(map(dict, chunked_data.rows[:]))
```

## Query Enhancement

In this section, we'll implement a query enhancement stage that improves our RAG pipeline. Our `QueryEnhancer` will perform three key tasks:

1. **Language Identification**: Detect whether the query is in English, Japanese, or Korean. This allows us to provide responses in the user's preferred language, enhancing the user experience.

2. **Intent Classification**: Determine if the query is relevant to our documentation. This helps filter out off-topic questions and allows for more appropriate responses.

3. **Sub-query Generation**: Break down complex queries into smaller, more focused sub-queries. This improves retrieval by capturing different aspects of the original question.

These enhancements serve two primary purposes:
- Inform the response generator, allowing it to tailor its output based on language and intent.
- Improve the retrieval process by using more targeted sub-queries.

Let's implement our `QueryEnhancer` and see it in action:


```python
from scripts.query_enhancer import QueryEnhancer
from scripts.utils import display_source

query_enhancer = QueryEnhancer()
```

## Analyzing Query Enhancement Results

Let's examine the output of our `QueryEnhancer` for the input: "How do I log images in lightning with wandb?"


```python
response = await query_enhancer.predict("How do I log images in lightning with wandb?")
response
```

Our `QueryEnhancer` has provided three key pieces of information:

1. **Language Detection**: The query is identified as English ('en'). This allows our system to respond in the appropriate language.

2. **Sub-query Generation**: The original query is broken down into more specific sub-queries:
   - "How to log images in lightning with wandb"
   - "How to log images in lightning"
   - "Log images wandb"
   - "Wandb image logging"
   - "Log images in lightning"
   
   These sub-queries help capture different aspects of the original question, potentially improving retrieval accuracy.

3. **Intent Classification**: The query is classified under the "integrations" intent. This suggests the user is asking about a specific integration between Lightning and Weights & Biases.

By leveraging this enhanced query information, our RAG system can now perform more targeted retrieval and generate more relevant, context-aware responses.

## Retriever: Leveraging BM25 from Previous Chapter

In our previous chapter, we explored the BM25 (Best Matching 25) retriever as an improvement over the basic TF-IDF approach. BM25 offers more nuanced rankings by incorporating document length normalization and term frequency saturation.

For this notebook, we'll continue using the same BM25 retriever. This consistency serves two important purposes:

1. **Fair Comparison**: By keeping the retriever constant, we can isolate the impact of our query enhancement techniques. This allows for a direct comparison between the QueryEnhancedRAGPipeline and the SimpleRAGPipeline.

2. **Leveraging Sub-queries**: Our new query-enhanced pipeline can take advantage of the sub-queries generated by the QueryEnhancer. We'll use these sub-queries to retrieve multiple context snippets, potentially providing more comprehensive information to the LLM.

Let's set up our BM25 retriever with the chunked data:


```python
from scripts.retriever import BM25Retriever

retriever = BM25Retriever()
retriever.index_data(chunked_data)
```

## Query-Enhanced Response Generation

With the additional information extracted from our query - specifically the language and intent - we can now create a more sophisticated response generator. The `QueryEnhancedResponseGenerator` class leverages this enriched context to produce more tailored and relevant responses.

Key Enhancements:
1. **Language-Aware Responses**: By incorporating the detected language, we can adjust the response style and potentially use language-specific resources or examples.

2. **Intent-Driven Generation**: The classified intent helps the model understand the user's goal, allowing for more focused and appropriate responses.

3. **Dynamic Prompt Engineering**: The system prompt is dynamically formatted with language and intent information, guiding the LLM's response generation process.

Let's examine the `QueryEnhancedResponseGenerator` class, paying special attention to how it utilizes the enhanced query information:


```python
from scripts.response_generator import QueryEnhanedResponseGenerator

display_source(QueryEnhanedResponseGenerator)
```

Note line 29 in the source code above. Here, we format the system prompt with the detected language and intents. This crucial step allows us to dynamically adapt our instructions to the LLM based on the specific query context.

By integrating these query enhancements into our response generation process, we create a more context-aware and adaptive RAG system. This approach should lead to more relevant, accurate, and tailored responses compared to our baseline system.

Building on our enhanced response generation, let's examine how these improvements are integrated into our overall RAG pipeline:

## Query-Enhanced RAG Pipeline

The `QueryEnhancedRAGPipeline` takes full advantage of our query enhancements, creating a more sophisticated and context-aware retrieval and generation process. Let's explore its key features:

1. **Multi-Query Retrieval**: The pipeline leverages the sub-queries generated by our `QueryEnhancer` to perform multiple retrieval operations. This broadens the scope of relevant information retrieved.

2. **Context Deduplication**: To optimize the input to our LLM, the pipeline deduplicates the retrieved chunks. This ensures we don't waste tokens on repetitive information.

3. **Intent-Based Flow Control**: The pipeline includes a crucial safety check based on the query's intent:

   ```python
   avoid_intents = ["unrelated", "needs_more_info", "opinion_request", "nefarious_query", "other"]
   
   for intent in intents:
       if intent["intent"] in avoid_intents:
           avoid_retrieval = True
           break
   ```

If the query's intent matches any in the `avoid_intents` list, the pipeline bypasses retrieval. This allows us to handle off-topic or inappropriate queries with pre-defined responses, enhancing the system's robustness and safety.

By integrating these features, our `QueryEnhancedRAGPipeline` creates a more flexible, efficient, and context-aware system compared to traditional RAG approaches. This should result in more relevant and appropriate responses across a wider range of query types.


```python
from scripts.rag_pipeline import QueryEnhancedRAGPipeline

display_source(QueryEnhancedRAGPipeline)
```

## Putting It All Together: Initializing and Testing Our Enhanced RAG System

Now that we've examined the components of our query-enhanced RAG system, let's bring everything together. We'll initialize the response generator with our new prompt, set up the complete RAG pipeline, and test it with a sample query. This will demonstrate how all the enhancements we've discussed work in concert to produce more relevant and context-aware responses.

Let's proceed step-by-step:


```python
# lets add the new prompt
QUERY_ENHANCED_PROMPT = open("prompts/query_enhanced_system.txt").read()

response_generator = QueryEnhanedResponseGenerator(
    model="command-r", prompt=QUERY_ENHANCED_PROMPT, client=cohere.AsyncClientV2()
)
```


```python
query_enhanced_rag_pipeline = QueryEnhancedRAGPipeline(
    query_enhancer=query_enhancer,
    retriever=retriever,
    response_generator=response_generator,
    top_k=2,
)

response = await query_enhanced_rag_pipeline.predict(
    "How do I log images in lightning with wandb?"
)


Markdown(response)
```

## Evaluate and Compare

## Evaluating Query Enhancement: Comparing Performance

Now that we've implemented our query-enhanced RAG system, it's crucial to quantify its performance improvements over our baseline. This evaluation will help us understand the impact of query enhancement on the overall RAG pipeline.

In this section, we'll:

1. Use the evaluation dataset from previous chapters to ensure consistency in our comparisons.
2. Focus on response quality metrics, as both systems use the same BM25 retriever.
3. Employ LLM-based metrics to assess the quality and relevance of generated responses.
4. Compare the performance of our QueryEnhancedRAGPipeline against the SimpleRAGPipeline.

By conducting this evaluation, we aim to answer key questions such as:
- Does query enhancement lead to more accurate and relevant responses?
- How does the system perform with different types of queries (e.g., simple vs. complex)?
- Are there specific areas where query enhancement shines or falls short?

We'll use Weave Evaluations to streamline our assessment process and visualize the results, providing clear insights into the effectiveness of our query enhancement techniques.

Let's begin by setting up our evaluation framework:


```python
eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

print(eval_dataset.rows[:2])
```


```python
# Let also initialize the baseline RAG pipeline from chapter 3

from scripts.rag_pipeline import SimpleRAGPipeline
from scripts.response_generator import SimpleResponseGenerator

INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)
simple_rag_pipeline = SimpleRAGPipeline(
    retriever=retriever, response_generator=response_generator, top_k=5
)
```

In this evaluation, we're primarily interested in assessing the overall response quality rather than individual retrieval metrics. Here's why:

1. **Common Retriever**: Both pipelines use the same BM25 retriever, making retrieval metric comparisons less informative.

2. **Enhanced Retrieval Process**: The query-enhanced pipeline retrieves more chunks due to its use of sub-queries, making direct retrieval comparisons potentially misleading.

3. **Holistic Evaluation**: Our goal is to understand the end-to-end performance improvement, which is best captured by analyzing the final output quality.

To achieve this, we'll employ LLM-based metrics from chapte 2 to evaluate response quality. These metrics provide a more nuanced and context-aware assessment of the generated answers, allowing us to:

- Gauge the relevance and accuracy of responses
- Determine if the enhanced pipeline better addresses user intents

By focusing on these response quality metrics, we can obtain a comprehensive view of how query enhancement impacts the overall performance of our RAG system.

Let's proceed with our evaluation:


```python
from scripts.response_metrics import LLM_METRICS

response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset,
    scorers=LLM_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)

baseline_response_scores = asyncio.run(
    response_evaluations.evaluate(simple_rag_pipeline)
)

query_enhanced_response_scores = asyncio.run(
    response_evaluations.evaluate(query_enhanced_rag_pipeline)
)
```

![compare_retriever_responses](../images/04_compare_query_enhanced_responses.png)

## Evaluation Results: Simple vs. Query-Enhanced RAG Pipelines

Our comparison between the SimpleRAGPipeline and QueryEnhancedRAGPipeline reveals interesting insights:

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

1. **Complexity vs. Performance**:
   Adding sophisticated features like query enhancement doesn't always lead to immediate improvements in automated metrics. It's crucial to balance complexity with measurable performance gains.

2. **Latency Considerations**:
   In real-world applications, the significant increase in latency could impact user experience. This highlights the importance of efficiency in RAG system design.

3. **Metric Limitations**:
   The discrepancy between perceived quality and metric scores underscores the limitations of current evaluation methods. This suggests an opportunity to fine-tune the LLM Judge for more nuanced assessments.

4. **Dataset Dependency**:
   These observations are specific to our current dataset. The benefits of query enhancement might vary across different types of queries or datasets. It's essential to evaluate RAG improvements on diverse, representative data.

5. **Qualitative vs. Quantitative Analysis**:
   This evaluation highlights the importance of combining automated metrics with qualitative analysis of individual responses for a comprehensive assessment.

6. **Iterative Improvement**:
   These results provide a foundation for further refinement. We might explore ways to optimize the query enhancement process, improve response conciseness, or refine our evaluation metrics.

This evaluation underscores the complexity of assessing RAG systems and the importance of multi-faceted evaluation approaches. It also demonstrates that improvements in system design may not always be immediately reflected in standard metrics, emphasizing the need for ongoing refinement of both RAG systems and evaluation methodologies.

## Key Takeaways

1. Query Enhancement Complexity: Advanced techniques like language identification, intent classification, and sub-query generation can significantly improve RAG system sophistication, but also introduce complexity.

2. Context-Aware Retrieval: Leveraging sub-queries and intent classification enables more comprehensive and targeted information retrieval.

3. Safety and Robustness: Intent-based flow control enhances system safety by appropriately handling off-topic or inappropriate queries.

4. Performance Trade-offs: Query enhancement can lead to more relevant responses but may increase latency. Consider the balance between sophistication and efficiency.

5. Evaluation Challenges: Standard metrics may not fully capture nuanced improvements in response quality. Combine quantitative and qualitative evaluation methods for comprehensive assessment.

6. Iterative Development: RAG system optimization is an ongoing process. Use evaluation results as a foundation for continuous refinement.

7. Holistic System Design: Effective RAG systems require careful integration of various components and consideration of their interactions.

8. Real-world Application: When implementing advanced RAG techniques, balance theoretical improvements with practical concerns like user experience and system complexity.
