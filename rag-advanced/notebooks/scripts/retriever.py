import bm25s
import Stemmer
import weave
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from .embedding import sync_embed
from .reranker import CohereReranker, FusionRanker
from typing import Callable
import numpy as np


class TFIDFRetriever(weave.Model):
    vectorizer: TfidfVectorizer = TfidfVectorizer()
    index: list = None
    data: list = None

    def index_data(self, data):
        self.data = data
        docs = [doc["cleaned_content"] for doc in data]
        self.index = self.vectorizer.fit_transform(docs)

    @weave.op()
    def search(self, query, k=5):
        query_vec = self.vectorizer.transform([query])
        cosine_distances = cdist(
            query_vec.todense(), self.index.todense(), metric="cosine"
        )[0]
        top_k_indices = cosine_distances.argsort()[:k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "source": self.data[idx]["metadata"]["source"],
                    "text": self.data[idx]["cleaned_content"],
                    "score": 1 - cosine_distances[idx],
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, k: int):
        return self.search(query, k)


stemmer: Stemmer.Stemmer = Stemmer.Stemmer("english")


class BM25Retriever(weave.Model):
    index: bm25s.BM25 = bm25s.BM25()
    data: list = None

    def index_data(self, data):
        self.data = data
        corpus = [doc["cleaned_content"] for doc in data]

        corpus_tokens = bm25s.tokenize(corpus)

        self.index.index(corpus_tokens, show_progress=False)

    @weave.op()
    def search(self, query, k=5):
        query_tokens = bm25s.tokenize(query)
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.index.retrieve(
            query_tokens, corpus=self.data, k=k, show_progress=False
        )

        output = []
        for idx in range(results.shape[1]):
            output.append(
                {
                    "source": results[0, idx]["metadata"]["source"],
                    "text": results[0, idx]["cleaned_content"],
                    "score": scores[0, idx],
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, k: int):
        return self.search(query, k)


class DenseRetriever(weave.Model):
    vectorizer: Callable = sync_embed
    index: np.ndarray = None
    data: list = None

    def index_data(self, data):
        self.data = data
        docs = [doc["cleaned_content"] for doc in data]
        embeddings = self.vectorizer(docs)
        self.index = np.array(embeddings)

    @weave.op()
    def search(self, query, k=5):
        query_embedding = self.vectorizer([query], input_type="search_query")
        cosine_distances = cdist(query_embedding, self.index, metric="cosine")[0]
        top_k_indices = cosine_distances.argsort()[:k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "source": self.data[idx]["metadata"]["source"],
                    "text": self.data[idx]["cleaned_content"],
                    "score": 1 - cosine_distances[idx],
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, k: int):
        return self.search(query, k)


class DenseRetrieverWithReranker(weave.Model):
    retriever: DenseRetriever = DenseRetriever()
    reranker: CohereReranker = CohereReranker()

    def index_data(self, data):
        self.retriever.index_data(data)

    @weave.op()
    def predict(self, query: str, top_k: int = None, top_n: int = None):
        if top_k and not top_n:
            top_n = top_k
            top_k = top_k * 2
        elif top_n and not top_k:
            top_k = top_n * 2
        else:
            top_k = 10
            top_n = 5
        retrievals = self.retriever.predict(query, top_k)
        reranked = self.reranker.predict(query, retrievals, top_n)
        return reranked


class HybridRetrieverReranker(weave.Model):
    sparse_retriever: BM25Retriever = BM25Retriever()
    dense_retriever: DenseRetrieverWithReranker = DenseRetrieverWithReranker()
    fusion_ranker: FusionRanker = FusionRanker()
    ranker: CohereReranker = CohereReranker()

    def index_data(self, data):
        self.sparse_retriever.index_data(data)
        self.dense_retriever.index_data(data)

    @weave.op()
    def predict(self, query: str, top_k: int = None, top_n: int = None):
        if top_k and not top_n:
            top_n = top_k
            top_k = top_k * 2
        elif top_n and not top_k:
            top_k = top_n * 2
        else:
            top_k = 10
            top_n = 5
        sparse_retrievals = self.sparse_retriever.predict(query, top_k)
        dense_retrievals = self.dense_retriever.predict(query, top_k)
        fused = self.fusion_ranker.predict(sparse_retrievals, dense_retrievals)
        reranked = self.ranker.predict(query, fused, top_n)
        return reranked
