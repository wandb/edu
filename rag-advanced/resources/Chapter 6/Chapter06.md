# Chapter 6

<a target="_blank" href="https://colab.research.google.com/github/wandb/edu/blob/main/rag-advanced/notebooks/Chapter06.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!--- @wandbcode{rag-course-06} -->

## Response Synthesis and Prompting

Response synthesis is a critical component of RAG systems, responsible for generating coherent and accurate answers based on retrieved information. In this chapter, we'll explore techniques to improve response quality through iterative prompt engineering and model selection.

Key concepts we'll cover:
1. Baseline prompt evaluation
2. Iterative prompt improvement
3. Impact of model selection on response quality
4. Comparative analysis of different prompting strategies

This hands-on experience will deepen your understanding of advanced RAG concepts and prepare you to implement these techniques in your own projects.

Let's begin by setting up our environment and importing the necessary libraries.

To begin, execute the following cell to clone the repository and install dependencies:


```
!git clone https://github.com/wandb/edu.git
%cd edu/rag-advanced
!pip install -qqq -r requirements.txt
%cd notebooks

import nltk

nltk.download("wordnet")
nltk.download("punkt_tab")
```

With the setup complete, we can now proceed with the chapter content.

Initial steps:
1. Log in to Weights & Biases (W&B)
2. Configure environment variables for API access

To obtain your Cohere API key, visit the [Cohere API dashboard](https://dashboard.cohere.com/api-keys).


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

# Data loading
We'll start by loading the semantically chunked data from Chapter 3. As a reminder, semantic chunking is an technique that groups related sentences together, preserving context and improving retrieval relevance.

This chunked data will serve as the input for the knowledge base for our RAG pipeline, allowing us to compare the effectiveness of our response synthesis techniques against a baseline system.

Let's load the data and take a look at the first few chunks:


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

Next, let's load the query enhancer, hybrid retriever, response generator and RAG pipeline from the previous chapters


```
import cohere

from scripts.query_enhancer import QueryEnhancer
from scripts.rag_pipeline import QueryEnhancedRAGPipeline
from scripts.response_generator import QueryEnhanedResponseGenerator
from scripts.retriever import HybridRetrieverReranker

query_enhancer = QueryEnhancer()
```

## Prompt iteration

Prompt engineering is a crucial skill in developing effective RAG systems. By carefully crafting prompts, we can guide the model to produce more accurate, relevant, and coherent responses. We'll explore several iterations of prompt improvements:

1. Baseline prompt
2. Adding precise instructions
3. Including response format examples
4. Incorporating model reasoning

For each iteration, we'll evaluate the impact on response quality using our established metrics.




```
eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

print(eval_dataset.rows[:2])
```


```
from scripts.response_metrics import ALL_METRICS as RESPONSE_METRICS

response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset,
    scorers=RESPONSE_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)
```


```
hybrid_retriever = HybridRetrieverReranker()
hybrid_retriever.index_data(chunked_data)
```

### Baseline Prompt Evaluation

We are now ready to evaluate the performance of the RAG pipeline while iterating over different prompt improvemtns.
For comparison, let's begin our evaluation of the baseline RAG pipeline.

This simple prompt serves as our starting point. It provides basic instructions for the model to answer questions about W&B using only the provided context. However, it lacks specific guidance on response structure, tone, or level of detail. As we iterate, we'll see how more detailed prompts can improve response quality and relevance.


```
INITIAL_PROMPT = open("prompts/initial_system.txt").read()

print(INITIAL_PROMPT)
```


```
baseline_response_generator = QueryEnhanedResponseGenerator(
    model="command-r", prompt=INITIAL_PROMPT, client=cohere.AsyncClient()
)


class BaselineRAGPipeline(QueryEnhancedRAGPipeline):
    pass


baseline_rag_pipeline = BaselineRAGPipeline(
    query_enhancer=query_enhancer,
    retriever=hybrid_retriever,
    response_generator=baseline_response_generator,
)


baseline_response_scores = asyncio.run(
    response_evaluations.evaluate(baseline_rag_pipeline)
)
```

**Tip**: When designing your initial prompt, aim for clarity and simplicity. However, be prepared to iterate and refine based on the results.

**Best Practice**: Always establish a baseline performance to measure improvements against.

### Improved Prompt V1: Adding Precise Instructions

In our first iteration, let's enhance the prompt by providing more detailed instructions to the AI assistant. We'll focus on:
1. Defining a clear role for the AI as a W&B specialist
2. Incorporating dynamic elements like language and intent recognition
3. Outlining a structured approach to formulating responses
4. Specifying formatting requirements, including markdown usage
5. Addressing edge cases, such as insufficient information or off-topic queries

By adding these elements, we aim to guide the model towards generating more coherent, relevant, and well-structured responses. This approach should help maintain accuracy while ensuring proper citation of sources. As we progress, we'll evaluate how these changes impact the quality of the generated answers.


```
# Can we improve the prompt with mode precise instructions ?

IMPROVED_PROMPT_V1 = open("prompts/improved_prompt_v1.txt").read()

print(IMPROVED_PROMPT_V1)
```

**Tip**: Adding specific instructions and defining the AI's role can significantly improve response quality.

**Best Practice**: Include guidelines for handling edge cases, such as insufficient information or off-topic queries, in your prompt design.


```
improved_v1_response_generator = QueryEnhanedResponseGenerator(
    model="command-r", prompt=IMPROVED_PROMPT_V1, client=cohere.AsyncClient()
)


class ImprovedV1RAGPipeline(QueryEnhancedRAGPipeline):
    pass


improved_v1_rag_pipeline = ImprovedV1RAGPipeline(
    query_enhancer=query_enhancer,
    retriever=hybrid_retriever,
    response_generator=improved_v1_response_generator,
)


improved_v1_response_scores = asyncio.run(
    response_evaluations.evaluate(improved_v1_rag_pipeline)
)
```

### Improved Prompt V2: Including Response Format Examples

In this iteration, we further refine our prompt by incorporating a concrete example of a well-structured response. This addition serves several purposes:

1. It demonstrates the desired formatting and structure, including proper use of markdown and code blocks.
2. It shows how to integrate citations and reference relevant documentation.
3. It illustrates the appropriate level of detail and explanation expected in responses.
4. It provides a model for balancing technical accuracy with user-friendly explanations.

By including this example, we aim to guide the model towards producing more consistent, well-formatted, and informative responses. This approach should help improve the overall quality and usefulness of the generated answers, making them more accessible to users with varying levels of technical expertise.


```
# Can we improve the prompt with a example of the response format ?

IMPROVED_PROMPT_V2 = open("prompts/improved_prompt_v2.txt").read()
print(IMPROVED_PROMPT_V2)
```

**Tip**: Providing concrete examples in your prompt can help guide the model towards the desired output format and structure.

**Best Practice**: When including examples, ensure they demonstrate key aspects like proper citation, use of markdown, and appropriate level of detail.


```
improved_v2_response_generator = QueryEnhanedResponseGenerator(
    model="command-r", prompt=IMPROVED_PROMPT_V2, client=cohere.AsyncClient()
)


class ImprovedV2RAGPipeline(QueryEnhancedRAGPipeline):
    pass


improved_v2_rag_pipeline = ImprovedV2RAGPipeline(
    query_enhancer=query_enhancer,
    retriever=hybrid_retriever,
    response_generator=improved_v2_response_generator,
)
improved_v2_response_scores = asyncio.run(
    response_evaluations.evaluate(improved_v2_rag_pipeline)
)
```

### Improved Prompt V3: Incorporating Model Reasoning

In this iteration, we focus on enhancing the model's reasoning process and transparency:

1. We introduce a structured approach to breaking down and addressing complex queries.
2. The prompt now explicitly requests the model to explain its thought process for each step.
3. We emphasize the importance of providing detailed explanations, including the relevance and functionality of code elements.
4. The example response demonstrates a clear, step-by-step structure with explanations at each stage.
5. We've added instructions for handling edge cases more comprehensively.

By encouraging the model to "show its work," we aim to produce more transparent, logical, and comprehensive responses. This approach can help users better understand the reasoning behind the answers, potentially leading to improved learning outcomes and increased trust in the AI assistant's capabilities. Additionally, this structured reasoning process may help the model catch and correct its own errors, leading to more accurate and reliable responses.


```
# Can we further improve the prompt to inlcude model reasoning ?


IMPROVED_PROMPT_V3 = open("prompts/improved_prompt_v3.txt").read()

print(IMPROVED_PROMPT_V3)
```

**Tip**: Encouraging the model to explain its reasoning process can lead to more transparent and logical responses.

**Best Practice**: Structure your prompt to guide the model through a step-by-step approach for complex queries.


```
improved_v3_response_generator = QueryEnhanedResponseGenerator(
    model="command-r", prompt=IMPROVED_PROMPT_V3, client=cohere.AsyncClient()
)


class ImprovedV3RAGPipeline(QueryEnhancedRAGPipeline):
    pass


improved_v3_rag_pipeline = ImprovedV3RAGPipeline(
    query_enhancer=query_enhancer,
    retriever=hybrid_retriever,
    response_generator=improved_v3_response_generator,
)

improved_v3_response_scores = asyncio.run(
    response_evaluations.evaluate(improved_v3_rag_pipeline)
)
```

### Model Improvement: Leveraging Advanced Language Models

After iterating on our prompt engineering, we now take the next step by utilizing a more advanced language model (command-r-plus). This change demonstrates an important principle in RAG system development: the synergy between prompt design and model capability. By combining our refined prompt with a more sophisticated model, we aim to:

1. Improve the overall quality and coherence of generated responses
2. Enhance the model's ability to understand and follow complex instructions
3. Potentially increase the accuracy and depth of domain-specific knowledge
4. Better handle nuanced queries and edge cases

This step allows us to explore how model selection interacts with prompt engineering to affect response quality. As we evaluate the results, we'll gain insights into the relative impact of prompt refinement versus model capabilities in our RAG pipeline.

**Tip**: Don't rely solely on prompt engineering; consider the capabilities of different models in your iterative improvement process.

**Best Practice**: Balance the trade-off between response quality and latency based on your specific use-case requirements.


```
# Can we further imporve by using a better model to generate the response ?

improved_v4_response_generator = QueryEnhanedResponseGenerator(
    model="command-r-plus", prompt=IMPROVED_PROMPT_V3, client=cohere.AsyncClient()
)


class ImprovedV4RAGPipeline(QueryEnhancedRAGPipeline):
    pass


improved_v4_rag_pipeline = ImprovedV4RAGPipeline(
    query_enhancer=query_enhancer,
    retriever=hybrid_retriever,
    response_generator=improved_v4_response_generator,
)

improved_v4_response_scores = asyncio.run(
    response_evaluations.evaluate(improved_v4_rag_pipeline)
)
```

## Comparing Evaluations

Comparing the performance of different RAG pipeline iterations is crucial for understanding the impact of our prompt engineering efforts. By comparing metrics across various versions, we can identify trends, improvements, and potential trade-offs. This comparative analysis helps us make informed decisions about which prompting strategies are most effective for our specific use case. It's important to consider both quantitative metrics (like accuracy scores) and qualitative aspects (such as response relevance) when assessing overall performance improvements.

**Tip**: Use multiple evaluation metrics to get a comprehensive view of your system's performance.

**Best Practice**: Regularly reassess and refine your prompts as you gather more data on user queries and system performance.

![compare_retriever_responses](../images/06_compare_prompts.png)

### Comparing RAG Pipeline Iterations

Here are a few key insights from the evaluation of the RAG pipeline iterations:

1. **Response Quality Improvement**: The ImprovedV3 pipelines significantly outperformed earlier versions in LLM Response Scorer metrics (0.95 vs 0.75 for baseline), indicating substantial improvements in response quality and correctness.

2. **Trade-off Between Quality and Latency**: While the later iterations (V3 and V4) produced higher quality responses, they also exhibited increased latency. This highlights a common trade-off in AI systems between performance and computational efficiency.

3. **Incremental Gains**: Each iteration showed improvements in various metrics, demonstrating the value of iterative refinement in prompt engineering and model selection.

4. **Metric Variability**: Some metrics (e.g., Levenshtein distance) showed unexpected increases in later iterations, reminding us that different evaluation metrics can capture different aspects of performance.

### Learnings

1. Prompt engineering can significantly impact response quality without changing the underlying model.
2. Combining refined prompts with more advanced models (as in V4) can lead to synergistic improvements.
3. The choice of evaluation metrics is crucial; a holistic view using multiple metrics provides a more comprehensive understanding of system performance.
4. In real-world applications, the balance between response quality and latency must be carefully considered based on specific use-case requirements.

This evaluation underscores the complexity of optimizing RAG systems and the importance of comprehensive, multi-faceted assessment approaches in AI development.


**Overall Best Practice**: "Iterative improvement is key in RAG system development. Continuously analyze results, gather feedback, and refine both prompts and model selection."

## Key Takeaways

1. Iterative Prompt Engineering: Systematic refinement of prompts can significantly enhance response quality without changing the underlying model.

2. Structured Instructions: Clear, detailed prompts with specific roles, formatting guidelines, and edge case handling improve response coherence and relevance.

3. Example Integration: Including well-crafted examples in prompts helps guide the model towards desired output structure and content quality.

4. Reasoning Transparency: Prompting the model to explain its thought process leads to more logical, comprehensive, and trustworthy responses.

5. Model-Prompt Synergy: Combining refined prompts with more advanced language models can yield synergistic improvements in response quality.

6. Performance Trade-offs: Higher quality responses often come at the cost of increased latency. Balance these factors based on specific use-case requirements.

7. Multifaceted Evaluation: Use a combination of metrics to comprehensively assess improvements, as different aspects of performance may not all improve uniformly.

8. Continuous Optimization: RAG system development is an ongoing process. Regularly reassess and refine prompts based on performance data and user feedback.

9. Scalability and Efficiency: As prompt complexity increases, consider the impact on system efficiency and scalability in real-world applications.
