"""
This module contains implementations of various retriever models for document retrieval.
"""
from typing import Callable

import bm25s
import numpy as np
import Stemmer
import weave
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

from .embedding import sync_embed
from .reranker import CohereReranker, FusionRanker


class TFIDFRetriever(weave.Model):
    """
    A retriever model that uses TF-IDF for indexing and searching documents.

    Attributes:
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
        index (list): The indexed data.
        data (list): The data to be indexed.
    """

    vectorizer: TfidfVectorizer = TfidfVectorizer()
    index: list = None
    data: list = None

    def index_data(self, data):
        """
        Indexes the provided data using TF-IDF.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self.data = data
        docs = [doc["cleaned_content"] for doc in data]
        self.index = self.vectorizer.fit_transform(docs)

    @weave.op()
    def search(self, query, k=5):
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        query_vec = self.vectorizer.transform([query])
        cosine_distances = cdist(
            query_vec.todense(), self.index.todense(), metric="cosine"
        )[0]
        top_k_indices = cosine_distances.argsort()[:k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "source": self.data.rows[idx]["metadata"]["source"],
                    "text": self.data.rows[idx]["cleaned_content"],
                    "score": 1 - cosine_distances[idx],
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, k: int):
        """
        Predicts the top-k results for the given query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        return self.search(query, k)


stemmer: Stemmer.Stemmer = Stemmer.Stemmer("english")


class BM25Retriever(weave.Model):
    """
    A retriever model that uses BM25 for indexing and searching documents.

    Attributes:
        index (bm25s.BM25): The BM25 index.
        data (list): The data to be indexed.
    """

    index: bm25s.BM25 = bm25s.BM25()
    data: list = None

    def index_data(self, data):
        """
        Indexes the provided data using BM25.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self.data = data
        corpus = [doc["cleaned_content"] for doc in data]

        corpus_tokens = bm25s.tokenize(corpus)

        self.index.index(corpus_tokens, show_progress=False)

    @weave.op()
    def search(self, query, k=5):
        """
        Searches the indexed data for the given query using BM25.

        Args:
            query (str): The search query.
            k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
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
        """
        Predicts the top-k results for the given query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        return self.search(query, k)


class DenseRetriever(weave.Model):
    """
    A retriever model that uses dense embeddings for indexing and searching documents.

    Attributes:
        vectorizer (Callable): The function used to generate embeddings.
        index (np.ndarray): The indexed embeddings.
        data (list): The data to be indexed.
    """

    vectorizer: Callable = sync_embed
    index: np.ndarray = None
    data: list = None

    def index_data(self, data):
        """
        Indexes the provided data using dense embeddings.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self.data = data
        docs = [doc["cleaned_content"] for doc in data]
        embeddings = self.vectorizer(docs)
        self.index = np.array(embeddings)

    @weave.op()
    def search(self, query, k=5):
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        query_embedding = self.vectorizer([query], input_type="search_query")
        cosine_distances = cdist(query_embedding, self.index, metric="cosine")[0]
        top_k_indices = cosine_distances.argsort()[:k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "source": self.data.rows[idx]["metadata"]["source"],
                    "text": self.data.rows[idx]["cleaned_content"],
                    "score": 1 - cosine_distances[idx],
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, k: int):
        """
        Predicts the top-k results for the given query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        return self.search(query, k)


class DenseRetrieverWithReranker(weave.Model):
    """
    A retriever model that uses dense embeddings for retrieval and a reranker for re-ranking the results.

    Attributes:
        retriever (DenseRetriever): The dense retriever model.
        reranker (CohereReranker): The reranker model.
    """

    retriever: DenseRetriever = DenseRetriever()
    reranker: CohereReranker = CohereReranker()

    def index_data(self, data):
        """
        Indexes the provided data using the dense retriever.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self.retriever.index_data(data)

    @weave.op()
    def predict(self, query: str, top_k: int = None, top_n: int = None):
        """
        Predicts the top-n results for the given query after re-ranking.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to retrieve before re-ranking. Default is None.
            top_n (int, optional): The number of top results to return after re-ranking. Default is None.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-n results.
        """
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
    """
    A hybrid retriever model that combines sparse and dense retrieval methods and uses a reranker for final ranking.

    Attributes:
        sparse_retriever (BM25Retriever): The sparse retriever model using BM25.
        dense_retriever (DenseRetrieverWithReranker): The dense retriever model with a reranker.
        fusion_ranker (FusionRanker): The fusion ranker to combine sparse and dense retrievals.
        ranker (CohereReranker): The final reranker model.
    """

    sparse_retriever: BM25Retriever = BM25Retriever()
    dense_retriever: DenseRetrieverWithReranker = DenseRetrieverWithReranker()
    fusion_ranker: FusionRanker = FusionRanker()
    ranker: CohereReranker = CohereReranker()

    def index_data(self, data):
        """
        Indexes the provided data using both sparse and dense retrievers.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self.sparse_retriever.index_data(data)
        self.dense_retriever.index_data(data)

    @weave.op()
    def predict(self, query: str, top_k: int = None, top_n: int = None):
        """
        Predicts the top-n results for the given query after re-ranking.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to retrieve before re-ranking. Default is None.
            top_n (int, optional): The number of top results to return after re-ranking. Default is None.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-n results.
        """
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
