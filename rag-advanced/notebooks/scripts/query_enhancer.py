"""
This module contains the QueryEnhancer class for enhancing user queries using LiteLLM.
"""
import json
import os
from enum import Enum
from typing import Any, Dict, List, Literal
import litellm
from litellm import acompletion
import weave
from ftlangdetect import detect as detect_language
from pydantic import BaseModel, Field
from instructor import from_litellm

from .utils import extract_json_from_markdown

completion_with_instructor = from_litellm(acompletion)

class Intent(BaseModel):
    """
    Model representing an intent with a label and a reason.
    """
    intent: Literal[
        "financial_performance",
        "operational_metrics",
        "market_analysis",
        "risk_assessment",
        "strategic_initiatives",
        "accounting_practices",
        "management_insights",
        "capital_structure",
        "segment_analysis",
        "comparative_analysis",
        "unrelated",
        "needs_more_info",
        "opinion_request",
        "nefarious_query",
        "other"
    ] = Field(
        ..., 
        description="The predicted intent label for the query"
    )
    reason: str = Field(
        ..., 
        description="The reasoning behind the predicted intent"
    )

class IntentPrediction(BaseModel):
    """
    Model representing a list of intents.
    """
    intents: List[Intent] = Field(
        ..., 
        description="List of predicted intents for the query"
    )

@weave.op()
async def parse_and_validate_response(response_text: str) -> Dict[str, Any]:
    """
    Parse and validate the response text.

    Args:
        response_text (str): The response text to be parsed and validated.

    Returns:
        Dict[str, Any]: A dictionary containing the validated response with enum keys replaced by their values.
    """

    # cleaned_text = extract_json_from_markdown(response_text)
    # print(cleaned_text)
    # parsed_response = json.loads(cleaned_text)
    # print(parsed_response)
    # validated_response = IntentPrediction(**parsed_response)
    validated_response = response_text
    response_dict = validated_response.model_dump()

    response_dict["intents"] = [
        {"intent": intent.intent, "reason": intent.reason}
        for intent in validated_response.intents
    ]

    return response_dict


@weave.op()
async def call_litellm_with_retry(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Call the LiteLLM API with retry logic.

    Args:
        messages (List[Dict[str, Any]]): The messages to send to the LiteLLM API.
        model (str, optional): The model to use. Defaults to "gpt-4o-mini".
        max_retries (int, optional): The maximum number of retries. Defaults to 5.

    Returns:
        Dict[str, Any]: The parsed and validated response from the LiteLLM API.

    Raises:
        Exception: If the maximum number of retries is reached without successful validation.
    """
    for attempt in range(max_retries):
        response_text = ""
        try:
            response = await completion_with_instructor.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
                response_model=IntentPrediction,
            )
            response_text = response

            return await parse_and_validate_response(response_text)
        except Exception as e:
            error_message = f"Your previous response resulted in an error: {str(e)}"
            error_message = (
                f"{error_message}\nPlease provide a valid JSON response based on the previous context and "
                f"error message. Ensure that:\n1. The response is a valid JSON object with an 'intents' "
                f"list.\n2. Each intent in the list has a valid 'intent' from the Labels enum and a "
                f"'reason'.\n3. The intents are unique.\n4. The response is not wrapped in markdown code "
                f"blocks."
            )

            if attempt == max_retries - 1:
                raise

            messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": error_message},
                ]
            )

    raise Exception("Max retries reached without successful validation")

class QueryEnhancer(weave.Model):
    """A class for enhancing user queries using LiteLLM."""

    @weave.op()
    async def generate_litellm_queries(self, query: str, model: str = "gpt-4o-mini") -> List[str]:
        """
        Generate search queries using LiteLLM.

        Args:
            query (str): The input query for which to generate search queries.
            model (str, optional): The model to use. Defaults to "gpt-4o-mini".

        Returns:
            List[str]: A list of generated search queries.
        """
        # load system prompt
        messages = json.load(open("prompts/search_query.json", "r"))
        # add user prompt (question)
        messages.append({"role": "user", "content": f"## Question\n{query}"})

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=500,
        )
        search_queries = response.choices[0].message.content.splitlines()
        return list(filter(lambda x: x.strip(), search_queries))

    @weave.op()
    async def get_intent_prediction(
        self,
        question: str,
        prompt_file: str = "prompts/intent_prompt.json",
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Get intent prediction for a given question using LiteLLM.

        Args:
            question (str): The question for which to get the intent prediction.
            prompt_file (str, optional): The file path to the prompt JSON. Defaults to "prompts/intent_prompt.json".
            model (str, optional): The model to use. Defaults to "gpt-4o-mini".

        Returns:
            Dict[str, Any]: A dictionary containing the intent predictions.
        """
        messages = json.load(open(prompt_file))
        messages.append(
            {"role": "user", "content": f"<question>\n{question}\n</question>\n"}
        )

        return await call_litellm_with_retry(messages, model)

    @weave.op()
    async def predict(self, query: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Predict the language, generate search queries, and get intent predictions for a given query.

        Args:
            query (str): The input query to process.
            model (str, optional): The model to use. Defaults to "gpt-4o-mini".

        Returns:
            Dict[str, Any]: A dictionary containing the original query, detected language, generated search queries, and intent predictions.
        """
        language = detect_language(query.replace("\n", " "))["lang"]
        search_queries = await self.generate_litellm_queries(query, model)
        intents = await self.get_intent_prediction(query, model=model)
        return {
            "query": query,
            "language": language,
            "search_queries": search_queries,
            "intents": intents["intents"],
        }
