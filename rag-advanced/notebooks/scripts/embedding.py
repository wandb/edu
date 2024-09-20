"""
This module provides functionality to embed texts using the Cohere API.
It includes an EmbeddingFunction class for asynchronous embedding and a sync_embed function for synchronous embedding.
"""

import asyncio
import os
from typing import List, Optional, Union

import cohere
from dotenv import load_dotenv

load_dotenv()

TextType = Union[str, List[str]]


class EmbeddingFunction:
    """
    A class to handle embedding functions using the Cohere API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = 50,
        model: str = "embed-english-v3.0",
    ):
        """
        Initialize the EmbeddingFunction.

        Args: api_key (Optional[str]): The API key for the Cohere API. If not provided, it will be fetched from the
        environment variable `CO_API_KEY`. batch_size (int): The number of texts to process in a single batch.
        Default is 50. model (str): The model to use for embedding. Default is "embed-english-v3.0".
        """
        self.api_key = api_key if api_key is not None else os.getenv("CO_API_KEY")
        self.client = cohere.AsyncClient(api_key=self.api_key)
        self.batch_size = batch_size
        self.embedding_model = model

    async def embed_batch(
        self, texts: TextType, input_type: str = "search_document"
    ) -> List[float]:
        """
        Embed a batch of texts.

        Args: texts (TextType): A single string or a list of strings to embed. input_type (str): The type of input,
        either "search_document" or "search_query". Default is "search_document".

        Returns:
            List[float]: A list of embeddings for the provided texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        response = await self.client.embed(
            texts=texts,
            model=self.embedding_model,
            input_type=input_type,
            embedding_types=["float"],
        )
        return response.embeddings.float

    async def embed_texts(
        self, texts: TextType, input_type: str = "search_document"
    ) -> List[float]:
        """
        Embed multiple texts, handling batching.

        Args: texts (TextType): A single string or a list of strings to embed. input_type (str): The type of input,
        either "search_document" or "search_query". Default is "search_document".

        Returns:
            List[float]: A list of embeddings for the provided texts.
        """
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
        """
        Embed a single query.

        Args:
            query (str): The query string to embed.

        Returns:
            List[float]: The embedding for the provided query.
        """
        return await self.embed_texts(query, input_type="search_query")

    async def embed_document(self, document: str) -> List[float]:
        """
        Embed a single document.

        Args:
            document (str): The document string to embed.

        Returns:
            List[float]: The embedding for the provided document.
        """
        return await self.embed_texts(document, input_type="search_document")

    async def __call__(
        self, texts: TextType, input_type: str = "search_document"
    ) -> List[float]:
        """
        Embed texts based on the input type.

        Args: texts (TextType): A single string or a list of strings to embed. input_type (str): The type of input,
        either "search_document" or "search_query". Default is "search_document".

        Returns:
            List[List[float]]: A list of embeddings for the provided texts.
        """
        if input_type == "search_query":
            return await self.embed_query(texts)
        else:
            return await self.embed_texts(texts, input_type=input_type)


def sync_embed(
    texts: TextType, input_type: str = "search_document"
) -> List[List[float]]:
    """
    Synchronously embed texts based on the input type.

    Args:
        texts (TextType): A single string or a list of strings to embed.
        input_type (str): The type of input, either "search_document" or "search_query". Default is "search_document".

    Returns:
        List[List[float]]: A list of embeddings for the provided texts.
    """
    embedding_function = EmbeddingFunction()
    return asyncio.get_event_loop().run_until_complete(
        embedding_function(texts, input_type=input_type)
    )
