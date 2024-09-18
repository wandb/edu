"""
This module contains classes for reranking documents using Cohere's reranking model and a fusion ranking approach.
"""
import json
import os
from typing import Any, Dict, List

import cohere
import numpy as np
import weave


class CohereReranker(weave.Model):
    """
    A class to rerank documents using Cohere's reranking model.
    """

    model: str = "rerank-english-v3.0"

    @weave.op()
    def rerank(self, query, docs, top_n=None):
        """
        Reranks the given documents based on their relevance to the query.

        Args:
            query (str): The query string.
            docs (List[Dict[str, Any]]): A list of documents to be reranked.
            top_n (int, optional): The number of top documents to return. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of reranked documents with relevance scores.
        """
        client = cohere.Client(os.environ["COHERE_API_KEY"])
        documents = [doc["text"] for doc in docs]
        response = client.rerank(
            model=self.model, query=query, documents=documents, top_n=top_n or len(docs)
        )

        outputs = []
        for doc in response.results:
            reranked_doc = docs[doc.index]
            reranked_doc["relevance_score"] = doc.relevance_score
            outputs.append(reranked_doc)
        return outputs[:top_n]

    @weave.op()
    def predict(self, query, docs, top_n=None):
        """
        Predicts the relevance of documents to the query by reranking them.

        Args:
            query (str): The query string.
            docs (List[Dict[str, Any]]): A list of documents to be reranked.
            top_n (int, optional): The number of top documents to return. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of reranked documents with relevance scores.
        """
        return self.rerank(query, docs, top_n)


class FusionRanker(weave.Model):
    """
    A class to rerank documents using a fusion ranking approach.
    """

    @weave.op()
    def rerank(self, *docs: List[List[Dict[Any, Any]]], k=60):
        """
        Reranks the given documents using a fusion ranking approach.

        Args:
            docs (List[List[Dict[Any, Any]]]): A variable number of lists of documents to be reranked.
            k (int, optional): A parameter to adjust the ranking score. Defaults to 60.

        Returns:
            List[Dict[Any, Any]]: A list of reranked documents with fusion scores.
        """

        class NumpyEncoder(json.JSONEncoder):
            """
            Custom JSON encoder for handling numpy data types.
            """

            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        fused_scores = {}
        for doc_list in docs:
            for rank, doc in enumerate(doc_list):
                doc_str = json.dumps(doc, cls=NumpyEncoder)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        reranked_results = []
        for doc, score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        ):
            doc = json.loads(doc)
            doc["fusion_score"] = score
            reranked_results.append(doc)

        return reranked_results

    @weave.op()
    def predict(self, *docs):
        """
        Predicts the relevance of documents by reranking them using the fusion ranking approach.

        Args:
            docs (List[List[Dict[Any, Any]]]): A variable number of lists of documents to be reranked.

        Returns:
            List[Dict[Any, Any]]: A list of reranked documents with fusion scores.
        """
        return self.rerank(*docs)
