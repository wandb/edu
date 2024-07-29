import asyncio
import os
from typing import List, Optional, Union

import cohere
from dotenv import load_dotenv

load_dotenv()

TextType = Union[str, List[str]]


class EmbeddingFunction:
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 50):
        self.api_key = api_key if api_key is not None else os.getenv("CO_API_KEY")
        self.client = cohere.AsyncClient(api_key=self.api_key)
        self.batch_size = batch_size

    async def embed_batch(
        self, texts: TextType, input_type: str = "search_document"
    ) -> List[float]:
        if isinstance(texts, str):
            texts = [texts]
        response = await self.client.embed(
            texts=texts, model="embed-english-v3.0", input_type=input_type
        )
        return response.embeddings

    async def embed_texts(
        self, texts: TextType, input_type: str = "search_document"
    ) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        tasks = [
            self.embed_batch(texts[i : i + self.batch_size], input_type=input_type)
            for i in range(0, len(texts), self.batch_size)
        ]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    async def embed_query(
        self,
        query: str,
    ) -> List[float]:
        return await self.embed_texts(query, input_type="search_query")

    async def embed_document(self, document: str) -> List[float]:
        return await self.embed_texts(document, input_type="search_document")

    async def __call__(
        self, texts: TextType, input_type: str = "search_document"
    ) -> List[List[float]]:
        if input_type == "search_query":
            return await self.embed_query(texts)
        else:
            return await self.embed_texts(texts, input_type=input_type)


def sync_embed(
    texts: TextType, input_type: str = "search_document"
) -> List[List[float]]:
    embedding_function = EmbeddingFunction()
    return asyncio.get_event_loop().run_until_complete(
        embedding_function(texts, input_type=input_type)
    )
