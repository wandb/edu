import weave
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

class Retriever(weave.Model):
    vectorizer: TfidfVectorizer = TfidfVectorizer()
    index: list = None
    data: list = None

    @weave.op()
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
