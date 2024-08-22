import os
from typing import Dict, List

import cohere # type: ignore
import weave

# Patch cohere to work with weave
from weave.integrations.cohere import cohere_patcher # type: ignore

cohere_patcher.attempt_patch()


class SimpleResponseGenerator(weave.Model):
    model: str
    prompt: str
    client: cohere.ClientV2 = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = cohere.ClientV2(
            api_key=os.environ["COHERE_API_KEY"],
            log_warning_experimental_features=False,
        )

    @weave.op()
    def generate_context_from_nodes(
        self, context: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        contexts = [
            {"source": item["source"], "text": item["text"]} for item in context
        ]
        return contexts

    def create_messages(self, query: str, context: List[Dict[str, any]]):
        _contexts = self.generate_context_from_nodes(context)
        contexts = [{"type": "text", "text": query}]
        for context in _contexts:
            contexts.append({"type": "document", "document": context})

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": contexts},
        ]
        return messages

    @weave.op()
    def generate_response(self, query: str, context: List[Dict[str, any]]) -> str:
        messages = self.create_messages(query, context)
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.1,
            max_tokens=2000,
        )
        return response.message.content[0].text

    @weave.op()
    def predict(self, query: str, context: List[Dict[str, any]]):
        return self.generate_response(query, context)


# TODO: update to cohere v2
class QueryEnhanedResponseGenerator(weave.Model):
    model: str
    prompt: str
    client: cohere.AsyncClient = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = cohere.AsyncClient(api_key=os.environ["CO_API_KEY"])

    @weave.op()
    def generate_context(self, context: List[Dict[str, any]]) -> List[Dict[str, any]]:
        _contexts = [
            {"source": item["source"], "text": item["text"]} for item in context
        ]
        contexts = []
        for context in _contexts:
            contexts.append({"type": "document", "document": context})
        return contexts

    @weave.op()
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, any]],
        language: str,
        intents: List[str],
    ) -> str:
        contexts = self.generate_context(context)
        response = await self.client.chat(
            preamble=self.prompt.format(language=language, intents=intents),
            message=query,
            model=self.model,
            documents=contexts,
            temperature=0.1,
            max_tokens=2000,
        )
        return response.text

    @weave.op()
    async def predict(
        self,
        query: str,
        context: List[Dict[str, any]],
        language: str,
        intents: List[str],
    ):
        return await self.generate_response(query, context, language, intents)
