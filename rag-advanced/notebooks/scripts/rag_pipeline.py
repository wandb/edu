import weave


class SimpleRAGPipeline(weave.Model):
    retriever: weave.Model = None
    response_generator: weave.Model = None
    top_k: int = 5

    @weave.op()
    def predict(self, query: str):
        context = self.retriever.predict(query, self.top_k)
        return self.response_generator.predict(query, context)
