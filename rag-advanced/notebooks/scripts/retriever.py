import bm25s
import Stemmer
import weave
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer


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

        self.index.index(corpus_tokens)

    @weave.op()
    def search(self, query, k=5):
        query_tokens = bm25s.tokenize(query)
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.index.retrieve(query_tokens, corpus=self.data, k=k)

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
