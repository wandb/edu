import weave
from typing import Optional, Union
from scripts.retriever import Retriever
from scripts.response_generator import ResponseGenerator


class RAGPipeline(weave.Model):
    retriever: Union[weave.Model, Retriever] = None
    response_generator: Union[weave.Model, ResponseGenerator] = None
    top_k: int = 5

    @weave.op()
    def predict(self, query: str):
        context = self.retriever.predict(query, self.top_k)
        return self.response_generator.predict(query, context)
