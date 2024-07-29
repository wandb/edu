import json
import os
from typing import Any, Dict, List

import cohere
import numpy as np
import weave
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from .utils import extract_json_from_markdown, make_cohere_api_call

load_dotenv()


@weave.op()
def compute_hit_rate(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the hit rate (precision) for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The hit rate (precision).

    The hit rate (precision) measures the proportion of retrieved documents that are relevant.
    It is calculated using the following formula:

    \[ \text{Hit Rate (Precision)} = \frac{\text{Number of Relevant Documents Retrieved}}{\text{Total Number of Documents Retrieved}} \]

    This metric is useful for assessing the accuracy of the retrieval system by determining the relevance of the retrieved documents.
    ```
    """
    search_results = [doc["source"] for doc in model_output]
    relevant_sources = [
        context["source"] for context in contexts if context["relevance"] != 0
    ]

    # Calculate the number of relevant documents retrieved
    relevant_retrieved = sum(
        1 for source in search_results if source in relevant_sources
    )

    # Calculate the hit rate (precision)
    hit_rate = relevant_retrieved / len(search_results) if search_results else 0.0

    return hit_rate


@weave.op
def compute_mrr(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The MRR score for the given query.

    MRR measures the rank of the first relevant document in the result list.
    It is calculated using the following formula:

    \[ \text{MRR} = \frac{1}{\text{rank of first relevant document}} \]

    If no relevant document is found, MRR is 0.

    This metric is useful for evaluating systems where there is typically one relevant document
    and the user is interested in finding that document quickly.
    """
    relevant_sources = [
        context["source"] for context in contexts if context["relevance"] != 0
    ]

    mrr_score = 0
    for rank, result in enumerate(model_output, 1):
        if result["source"] in relevant_sources:
            mrr_score += 1 / rank

    if mrr_score == 0:
        return 0.0
    else:
        return mrr_score / len(model_output)


# NDCG (Normalized Discounted Cumulative Gain)
@weave.op
def compute_ndcg(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.
                - 'relevance': The relevance score of the document (0, 1, or 2).

    Returns:
        float: The NDCG score for the given query.
    """
    # Create a mapping of source to relevance
    relevance_map = {context["source"]: context["relevance"] for context in contexts}

    dcg = 0.0
    idcg = 0.0

    # Calculate DCG
    for i, result in enumerate(model_output):
        rel = relevance_map.get(result["source"], 0)
        dcg += (2**rel - 1) / np.log2(i + 2)

    # Calculate IDCG
    sorted_relevances = sorted(
        [context["relevance"] for context in contexts], reverse=True
    )
    for i, rel in enumerate(sorted_relevances):
        idcg += (2**rel - 1) / np.log2(i + 2)

    # To avoid division by zero
    if idcg == 0:
        return 0.0

    # Calculate nDCG
    ndcg = dcg / idcg
    return ndcg


# MAP (Mean Average Precision)
@weave.op()
def compute_map(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the Mean Average Precision (MAP) for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The MAP score for the given query.

    MAP provides a single-figure measure of quality across recall levels.
    For a single query, it's equivalent to the Average Precision (AP).
    It's calculated using the following formula:

    \[ \text{MAP} = \frac{\sum_{k=1}^n P(k) \times \text{rel}(k)}{\text{number of relevant documents}} \]

    Where:
    - n is the number of retrieved documents
    - P(k) is the precision at cut-off k in the list
    - rel(k) is an indicator function: 1 if the item at rank k is relevant, 0 otherwise
    MAP considers both precision and recall, as well as the ranking of relevant documents.

    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }

    num_relevant = 0
    sum_precision = 0.0

    for i, result in enumerate(model_output):
        if result["source"] in relevant_sources:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    average_precision = sum_precision / len(relevant_sources)
    return average_precision


@weave.op()
def compute_precision(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the Precision for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The Precision score for the given query.

    Precision measures the proportion of retrieved documents that are relevant.
    It is calculated using the following formula:

    \[ \text{Precision} = \frac{\text{Number of Relevant Documents Retrieved}}{\text{Total Number of Documents Retrieved}} \]
    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }
    retrieved_sources = {result["source"] for result in model_output}

    relevant_retrieved = relevant_sources & retrieved_sources

    precision = (
        len(relevant_retrieved) / len(retrieved_sources) if retrieved_sources else 0.0
    )
    return precision


# Recall
@weave.op()
def compute_recall(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the Recall for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The Recall score for the given query.

    Recall measures the proportion of relevant documents that are retrieved.
    It is calculated using the following formula:

    \[ \text{Recall} = \frac{\text{Number of Relevant Documents Retrieved}}{\text{Total Number of Relevant Documents}} \]
    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }
    retrieved_sources = {result["source"] for result in model_output}

    relevant_retrieved = relevant_sources & retrieved_sources

    recall = (
        len(relevant_retrieved) / len(relevant_sources) if relevant_sources else 0.0
    )
    return recall


# F1 Score
@weave.op()
def compute_f1_score(
    model_output: List[Dict[str, Any]], contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate the F1-Score for a single query.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
            Each dictionary contains:
                - 'source': A unique identifier for the document.
                - 'score': The relevance score of the document.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The F1-Score for the given query.

    F1-Score is the harmonic mean of Precision and Recall.
    It is calculated using the following formula:

    \[ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]
    """
    precision = compute_precision(model_output, contexts)
    recall = compute_recall(model_output, contexts)

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


@weave.op()
async def parse_and_validate_response(
    response_text: str, num_contexts: int
) -> Dict[str, Any]:
    """Parse and validate the response text."""

    class RelevanceScore(BaseModel):
        id: int
        relevance: int

        @field_validator("relevance")
        def check_relevance_range(cls, v):
            if v not in [0, 1, 2]:
                raise ValueError(f"Relevance must be 0, 1, or 2. Got {v}")
            return v

    class RelevanceResponse(BaseModel):
        final_scores: List[RelevanceScore]

        @field_validator("final_scores")
        def check_unique_ids(cls, v, values, **kwargs):
            ids = [score.id for score in v]
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")
            return v

    cleaned_text = extract_json_from_markdown(response_text)
    parsed_response = json.loads(cleaned_text)
    validated_response = RelevanceResponse(**parsed_response)

    if len(validated_response.final_scores) != num_contexts:
        raise ValueError(
            f"Expected {num_contexts} scores, but got {len(validated_response.final_scores)}"
        )

    return validated_response.model_dump()


@weave.op()
async def call_cohere_with_retry(
    co_client: cohere.AsyncClient,
    preamble: str,
    chat_history: List[Dict[str, str]],
    message: str,
    num_contexts: int,
    max_retries: int = 5,
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        response_text = ""
        try:
            response_text = await make_cohere_api_call(
                co_client,
                preamble,
                chat_history,
                message,
                model="command-r-plus",
                force_single_step=True,
                temperature=0.0,
                prompt_truncation="AUTO",
                max_tokens=250,
            )

            return await parse_and_validate_response(response_text, num_contexts)
        except Exception as e:
            error_message = f"Your previous response resulted in an error: {str(e)}"

            if attempt == max_retries - 1:
                raise

            chat_history.extend(
                [
                    {"role": "USER", "message": message},
                    {"role": "CHATBOT", "message": response_text},
                ]
            )
            message = f"{error_message}\nPlease provide a valid JSON response based on the previous context and error message. Ensure that:\n1. The number of scores matches the number of contexts ({num_contexts}).\n2. The IDs are unique.\n3. The relevance scores are 0, 1, or 2.\n4. The response is a valid JSON object, not wrapped in markdown code blocks."

    raise Exception("Max retries reached without successful validation")


@weave.op()
async def evaluate_retrieval_with_llm(
    question: str,
    contexts: List[Dict[str, Any]],
    prompt_file: str = "prompts/retrieval_eval.json",
) -> Dict[str, Any]:
    co_client = cohere.AsyncClient(api_key=os.environ["CO_API_KEY"])

    # Load the prompt
    messages = json.load(open(prompt_file))
    preamble = messages[0]["message"]
    chat_history = messages[1:]

    # Prepare the message
    message_template = """<question>
    {question}
    </question>
    {context}
    """
    context = ""
    for idx, doc in enumerate(contexts):
        context += f"<doc_{idx}>\n{doc['text']}\n</doc_{idx}>\n"
    message = message_template.format(question=question, context=context)

    # Make the API call with retry logic
    return await call_cohere_with_retry(
        co_client, preamble, chat_history, message, len(contexts)
    )


@weave.op()
def compute_rank_score(scores: List[int]) -> float:
    rank_score = 0
    for rank, result in enumerate(scores, 1):
        if result == 2:
            rank_score = 1 / rank
            return rank_score
    return rank_score


@weave.op()
async def llm_retrieval_scorer(
    model_output: Dict[str, Any], question: str
) -> Dict[str, float]:
    scores = await evaluate_retrieval_with_llm(question, model_output)
    relevance_scores = [item["relevance"] for item in scores["final_scores"]]
    mean_relevance = sum(relevance_scores) / len(model_output)
    rank_score = compute_rank_score(relevance_scores)
    return {"relevance": mean_relevance, "relevance_rank_score": rank_score}


IR_METRICS = [
    compute_hit_rate,
    compute_mrr,
    compute_ndcg,
    compute_map,
    compute_precision,
    compute_recall,
    compute_f1_score,
]

LLM_METRICS = [
    llm_retrieval_scorer,
]

ALL_METRICS = IR_METRICS + LLM_METRICS
