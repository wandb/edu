"""
This module contains operations for evaluating retrieval results using various metrics.
"""
import json
import os
from typing import Any, Dict, List

import cohere
import numpy as np
import weave
from pydantic import BaseModel, field_validator

from .utils import extract_json_from_markdown, make_cohere_api_call


@weave.op
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

    This metric is useful for assessing the accuracy of the retrieval system by determining the relevance of the
    retrieved documents.
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
    relevance_map = {context["source"]: context["relevance"] for context in contexts}

    dcg = 0.0
    idcg = 0.0

    for i, result in enumerate(model_output):
        rel = relevance_map.get(result["source"], 0)
        dcg += (2**rel - 1) / np.log2(i + 2)

    sorted_relevances = sorted(
        [context["relevance"] for context in contexts], reverse=True
    )
    for i, rel in enumerate(sorted_relevances):
        idcg += (2**rel - 1) / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


@weave.op
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


@weave.op
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
@weave.op
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
@weave.op
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
    """
    precision = compute_precision(model_output, contexts)
    recall = compute_recall(model_output, contexts)

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


@weave.op
async def parse_and_validate_response(
    response_text: str, num_contexts: int
) -> Dict[str, Any]:
    """
    Parse and validate the response text.

    Args:
        response_text (str): The response text to be parsed and validated.
        num_contexts (int): The expected number of contexts.

    Returns:
        Dict[str, Any]: The validated response as a dictionary.

    Raises:
        ValueError: If the relevance score is not 0, 1, or 2.
        ValueError: If the IDs are not unique.
        ValueError: If the number of scores does not match the number of contexts.
    """

    class RelevanceScore(BaseModel):
        """
        A model representing a relevance score for a document.

        Attributes:
            id (int): The unique identifier for the document.
            relevance (int): The relevance score of the document (0, 1, or 2).
        """

        id: int
        relevance: int

        @field_validator("relevance")
        def check_relevance_range(cls, v):
            """
            Validate that the relevance score is within the acceptable range.

            Args:
                v (int): The relevance score to validate.

            Returns:
                int: The validated relevance score.

            Raises:
                ValueError: If the relevance score is not 0, 1, or 2.
            """
            if v not in [0, 1, 2]:
                raise ValueError(f"Relevance must be 0, 1, or 2. Got {v}")
            return v

    class RelevanceResponse(BaseModel):
        """
        A model representing the response containing relevance scores for documents.

        Attributes:
            final_scores (List[RelevanceScore]): A list of relevance scores for the documents.
        """

        final_scores: List[RelevanceScore]

        @field_validator("final_scores")
        def check_unique_ids(cls, v, values, **kwargs):
            """
            Validate that the IDs in the final_scores list are unique.

            Args:
                v (List[RelevanceScore]): The list of relevance scores to validate.
                values (dict): Additional values passed to the validator.
                **kwargs: Additional keyword arguments.

            Returns:
                List[RelevanceScore]: The validated list of relevance scores.

            Raises:
                ValueError: If the IDs in the final_scores list are not unique.
            """
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


@weave.op
async def call_cohere_with_retry(
    co_client: cohere.AsyncClientV2,
    messages: List[Dict[str, any]],
    num_contexts: int,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Call the Cohere API with retry logic.

    Args:
        co_client (cohere.AsyncClientV2): The Cohere client instance.
        messages (List[Dict[str, any]]): The list of messages to send to the API.
        num_contexts (int): The expected number of contexts.
        max_retries (int, optional): The maximum number of retry attempts. Defaults to 5.

    Returns:
        Dict[str, Any]: The validated response from the API.

    Raises:
        Exception: If the maximum number of retries is reached without successful validation.
    """
    for attempt in range(max_retries):
        response_text = ""
        try:
            response_text = await make_cohere_api_call(
                co_client,
                messages,
                model="command-r-plus",
                temperature=0.0,
                max_tokens=250,
            )
            return await parse_and_validate_response(response_text, num_contexts)
        except Exception as e:
            error_message = f"Your previous response resulted in an error: {str(e)}"
            error_message = (
                f"{error_message}\nPlease provide a valid JSON response based on the previous context and "
                f"error message. Ensure that:\n1. The number of scores matches the number of contexts ({num_contexts})."
                f"\n2. The IDs are unique.\n3. The relevance scores are 0, 1, or 2.\n4. The"
                f"response is a valid JSON object, not wrapped in markdown code blocks."
            )

            if attempt == max_retries - 1:
                raise

            messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": error_message},
                ]
            )

    raise Exception("Max retries reached without successful validation")


@weave.op
async def evaluate_retrieval_with_llm(
    question: str,
    contexts: List[Dict[str, Any]],
    prompt_file: str = "prompts/retrieval_eval.json",
) -> Dict[str, Any]:
    """
    Evaluate the retrieval results using a language model.

    Args:
        question (str): The query or question for which the retrieval is being evaluated.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the retrieved documents.
        prompt_file (str, optional): The file path to the prompt template. Defaults to "prompts/retrieval_eval.json".

    Returns:
        Dict[str, Any]: The validated response from the language model.
    """
    co_client = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])

    messages = json.load(open(prompt_file))

    message_template = """<question>
    {question}
    </question>
    {context}
    """
    context = ""
    for idx, doc in enumerate(contexts):
        context += f"<doc_{idx}>\n{doc['text']}\n</doc_{idx}>\n"

    messages.append(
        {
            "role": "user",
            "content": message_template.format(question=question, context=context),
        }
    )

    return await call_cohere_with_retry(co_client, messages, len(contexts))


@weave.op
def compute_rank_score(scores: List[int]) -> float:
    """
    Calculate the rank score for a list of relevance scores.

    Args:
        scores (List[int]): A list of relevance scores where 2 indicates the highest relevance.

    Returns:
        float: The rank score, which is the reciprocal of the rank of the first highly relevant document (score of 2).
               If no such document is found, returns 0.
    """
    rank_score = 0
    for rank, result in enumerate(scores, 1):
        if result == 2:
            rank_score = 1 / rank
            return rank_score
    return rank_score


@weave.op
async def llm_retrieval_scorer(
    model_output: List[Dict[str, Any]], question: str
) -> Dict[str, float]:
    """
    Evaluate the retrieval results using a language model and compute relevance scores.

    Args:
        model_output (List[Dict[str, Any]]): The list of retrieved documents from the model.
        question (str): The query or question for which the retrieval is being evaluated.

    Returns:
        Dict[str, float]: A dictionary containing the mean relevance score and the relevance rank score.
    """
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
