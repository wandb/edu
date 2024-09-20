"""
A module containing response generators using Cohere's API for generating responses.
"""

import os
from typing import Dict, List

import cohere
import weave
from weave.integrations.cohere import cohere_patcher

cohere_patcher.attempt_patch()


class SimpleResponseGenerator(weave.Model):
    """
    A simple response generator model using Cohere's API.

    Attributes:
        model (str): The model name to be used for generating responses.
        prompt (str): The prompt to be used for generating responses.
        client (cohere.ClientV2): The Cohere client for interacting with the Cohere API.
    """

    model: str
    prompt: str
    client: cohere.ClientV2 = None

    def __init__(self, **kwargs):
        """
        Initialize the SimpleResponseGenerator with the provided keyword arguments.
        Sets up the Cohere client using the API key from environment variables.
        """
        super().__init__(**kwargs)
        self.client = cohere.ClientV2(
            api_key=os.environ["COHERE_API_KEY"],
        )

    @weave.op()
    def generate_context(self, context: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Generate a list of contexts from the provided context list.

        Args:
            context (List[Dict[str, any]]): A list of dictionaries containing context data.

        Returns:
            List[Dict[str, any]]: A list of dictionaries with 'source' and 'text' keys.
        """
        contexts = [
            {"source": item["source"], "text": item["text"]} for item in context
        ]
        return contexts

    def create_messages(self, query: str, context: List[Dict[str, any]]):
        """
        Create a list of messages for the chat model based on the query and context.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.

        Returns:
            List[Dict[str, any]]: A list of messages formatted for the chat model.
        """
        documents = self.generate_context(context)
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": query},
        ]
        documents = documents
        return messages

    @weave.op()
    def generate_response(self, query: str, context: List[Dict[str, any]]) -> str:
        """
        Generate a response from the chat model based on the query and context.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.

        Returns:
            str: The generated response from the chat model.
        """
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
        """
        Predict the response for the given query and context.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.

        Returns:
            str: The predicted response from the chat model.
        """
        return self.generate_response(query, context)


class QueryEnhanedResponseGenerator(weave.Model):
    """
    A response generator model that enhances queries with additional context, language, and intents.

    Attributes:
        model (str): The model name to be used for generating responses.
        prompt (str): The prompt to be used for generating responses.
        client (cohere.AsyncClient): The asynchronous Cohere client for interacting with the Cohere API.
    """

    model: str
    prompt: str
    client: cohere.AsyncClient = None

    def __init__(self, **kwargs):
        """
        Initialize the QueryEnhanedResponseGenerator with the provided keyword arguments.
        Sets up the asynchronous Cohere client using the API key from environment variables.
        """
        super().__init__(**kwargs)
        self.client = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])

    @weave.op()
    def generate_context(self, context: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Generate a list of contexts from the provided context list.

        Args:
            context (List[Dict[str, any]]): A list of dictionaries containing context data.

        Returns:
            List[Dict[str, any]]: A list of dictionaries with 'source' and 'text' keys.
        """
        contexts = [
            {"data": {"source": item["source"], "text": item["text"]}}
            for item in context
        ]
        return contexts

    def create_messages(
        self,
        query: str,
        context: List[Dict[str, any]],
        language: str,
        intents: List[str],
    ):
        """
        Create a list of messages for the chat model based on the query, context, language, and intents.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.
            language (str): The language to be used in the response.
            intents (List[str]): A list of intents to be considered in the response.

        Returns:
            List[Dict[str, any]]: A list of messages formatted for the chat model.
        """
        documents = self.generate_context(context)

        messages = [
            {
                "role": "system",
                "content": self.prompt.format(language=language, intents=intents),
            },
            {"role": "user", "content": query},
        ]
        documents = documents
        return messages

    @weave.op()
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, any]],
        language: str,
        intents: List[str],
    ) -> str:
        """
        Generate a response from the chat model based on the query, context, language, and intents.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.
            language (str): The language to be used in the response.
            intents (List[str]): A list of intents to be considered in the response.

        Returns:
            str: The generated response from the chat model.
        """
        messages = self.create_messages(query, context, language, intents)
        response = await self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.1,
            max_tokens=2000,
        )
        return response.message.content[0].text

    @weave.op()
    async def predict(
        self,
        query: str,
        context: List[Dict[str, any]],
        language: str,
        intents: List[str],
    ):
        """
        Predict the response for the given query, context, language, and intents.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.
            language (str): The language to be used in the response.
            intents (List[str]): A list of intents to be considered in the response.

        Returns:
            str: The predicted response from the chat model.
        """
        return await self.generate_response(query, context, language, intents)
