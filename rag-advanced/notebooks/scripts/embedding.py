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

    async def embed_batch(self, texts: TextType) -> List[float]:
        if isinstance(texts, str):
            texts = [texts]
        response = await self.client.embed(
            texts=texts, model="embed-english-v3.0", input_type="search_document"
        )
        return response.embeddings

    async def embed_texts(self, texts: TextType) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        tasks = [
            self.embed_batch(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    async def __call__(self, texts: TextType) -> List[List[float]]:
        return await self.embed_texts(texts)


def sync_embed(texts: TextType) -> List[List[float]]:
    embedding_function = EmbeddingFunction()
    return asyncio.get_event_loop().run_until_complete(embedding_function(texts))
