import os
import cohere
import weave
import json
from typing import List, Dict, Any
import numpy as np

from dotenv import load_dotenv

load_dotenv()


class CohereReranker(weave.Model):
    model: str = "rerank-english-v3.0"

    @weave.op()
    def rerank(self, query, docs, top_n=None):
        client = cohere.Client(os.environ["CO_API_KEY"])
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
        return self.rerank(query, docs, top_n)


class FusionRanker(weave.Model):

    @weave.op()
    def rerank(self, *docs: List[List[Dict[Any, Any]]], k=60):

        class NumpyEncoder(json.JSONEncoder):
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
        return self.rerank(*docs)
