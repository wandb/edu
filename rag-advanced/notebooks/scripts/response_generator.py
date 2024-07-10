import os
from typing import List, Dict
import cohere
import weave

class ResponseGenerator(weave.Model):
    model: str
    prompt: str
    client: cohere.Client = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = cohere.Client(api_key=os.environ["CO_API_KEY"])

    @weave.op()
    def generate_context(self, context: List[Dict[str, any]]) -> str:
        return [{"source": item['source'], "text": item['text']} for item in context]
    
    @weave.op()
    def generate_response(self, query: str, context: List[Dict[str, any]]) -> str:
        contexts = self.generate_context(context)
        response = self.client.chat(
            preamble=self.prompt,
            message=query,
            model=self.model,
            documents=contexts,
            temperature=0.1,
            max_tokens=2000,
        )
        return response.text

    @weave.op()
    def predict(self, query: str, context: List[Dict[str, any]]):
        return self.generate_response(query, context)
