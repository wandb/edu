import weave
import numpy as np
from typing import List, Dict, Any


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
            mrr_score = 1 / rank
            break
    return mrr_score


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
                - 'score': The cosine similarity score of the document to the query.
        contexts (List[Dict[str, Any]]): A list of dictionaries representing the relevant contexts.
            Each dictionary contains:
                - 'source': A unique identifier for the relevant document.

    Returns:
        float: The NDCG score for the given query.

    NDCG measures the ranking quality of the search results, taking into account the position of relevant documents.

    NDCG Formula:
    1. Calculate the Discounted Cumulative Gain (DCG):
       \[ \text{DCG}_p = \sum_{i=1}^p \frac{2^{rel_i} - 1}{\log_2(i + 1)} \]
       where \( rel_i \) is the relevance score of the document at position \( i \).

    2. Calculate the Ideal Discounted Cumulative Gain (IDCG), which is the DCG of the ideal ranking:
       \[ \text{IDCG}_p = \sum_{i=1}^p \frac{2^{rel_i} - 1}{\log_2(i + 1)} \]
       where documents are sorted by their relevance scores in descending order.

    3. Normalize the DCG by dividing it by the IDCG to get NDCG:
       \[ \text{NDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p} \]

    This implementation uses continuous relevance scores.
    """
    relevant_sources = {
        context["source"] for context in contexts if context["relevance"] != 0
    }

    dcg = 0.0
    idcg = 0.0

    # Calculate DCG
    for i, result in enumerate(model_output):
        if result["source"] in relevant_sources:
            dcg += (2 ** result["score"] - 1) / np.log2(
                i + 2
            )  # i+2 because log2 starts at 1 for i=0

    # Sort the results by score to calculate IDCG
    sorted_model_output = sorted(model_output, key=lambda x: x["score"], reverse=True)

    # Calculate IDCG
    for i, result in enumerate(sorted_model_output):
        if result["source"] in relevant_sources:
            idcg += (2 ** result["score"] - 1) / np.log2(i + 2)

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
