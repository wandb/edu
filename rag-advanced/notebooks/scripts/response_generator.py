"""
A module containing response generators using LiteLLM for generating responses.
"""

import os
from typing import Dict, List

import weave
from litellm import completion, acompletion


class SimpleResponseGenerator(weave.Model):
    """
    A simple response generator model using LiteLLM.

    Attributes:
        model (str): The model name to be used for generating responses.
        prompt (str): The prompt to be used for generating responses.
    """

    model: str
    prompt: str

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

    def create_messages(self, query: str, context: List[Dict[str, any]]):
        """
        Create a list of messages for the chat model based on the query and context.

        Args:
            query (str): The user's query.
            context (List[Dict[str, any]]): A list of dictionaries containing context data.

        Returns:
            List[Dict[str, any]]: A list of messages formatted for the chat model.
        """
        messages = [
            {"role": "system", "content": self.prompt},
        ]
        
        if not self.model.startswith("cohere"):
            formatted_context = "\n\n".join([f"Source: {item['data']['source']}\nText: {item['data']['text']}" for item in context])
            messages.append({"role": "user", "content": f"Context:\n{formatted_context}\n\nQuery: {query}"})
        else:
            messages.append({"role": "user", "content": query})
        
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
        documents = self.generate_context(context)  
        messages = self.create_messages(query, documents)
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        
        if self.model.startswith("cohere"):
            kwargs["documents"] = documents
        
        response = completion(**kwargs)
        return response['choices'][0]['message']['content']

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


class QueryEnhancedResponseGenerator(weave.Model):
    """
    A response generator model that enhances queries with additional language and intents.

    Attributes:
        model (str): The model name to be used for generating responses.
        prompt (str): The prompt to be used for generating responses.
    """

    model: str
    prompt: str

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
        messages = [
            {
                "role": "system",
                "content": self.prompt.format(language=language, intents=intents),
            },
        ]
        
        if not self.model.startswith("cohere"):
            formatted_context = "\n\n".join([f"Source: {item['data']['source']}\nText: {item['data']['text']}" for item in context])
            messages.append({"role": "user", "content": f"Context:\n{formatted_context}\n\nQuery: {query}"})
        else:
            messages.append({"role": "user", "content": query})
        
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
        documents = self.generate_context(context)
        messages = self.create_messages(query, documents, language, intents)
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        
        if self.model.startswith("cohere"):
            kwargs["documents"] = documents
        
        response = await acompletion(**kwargs)
        return response['choices'][0]['message']['content']

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
