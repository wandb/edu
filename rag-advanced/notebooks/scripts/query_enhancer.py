"""
This module contains the QueryEnhancer class for enhancing user queries using the Cohere API.
"""
import json
import os
from enum import Enum
from typing import Any, Dict, List

import cohere
import weave
from ftlangdetect import detect as detect_language
from pydantic import BaseModel

from .utils import extract_json_from_markdown, make_cohere_api_call


@weave.op()
@weave.op()
async def parse_and_validate_response(response_text: str) -> Dict[str, Any]:
    """
    Parse and validate the response text.

    Args:
        response_text (str): The response text to be parsed and validated.

    Returns:
        Dict[str, Any]: A dictionary containing the validated response with enum keys replaced by their values.
    """

    class Labels(str, Enum):
        """
        Enum representing different intent labels.
        """

        UNRELATED = "unrelated"
        CODE_TROUBLESHOOTING = "code_troubleshooting"
        INTEGRATIONS = "integrations"
        PRODUCT_FEATURES = "product_features"
        SALES_AND_GTM_RELATED = "sales_and_gtm_related"
        BEST_PRACTICES = "best_practices"
        COURSE_RELATED = "course_related"
        NEEDS_MORE_INFO = "needs_more_info"
        OPINION_REQUEST = "opinion_intent_promptrequest"
        NEFARIOUS_QUERY = "nefarious_query"
        OTHER = "other"

    class Intent(BaseModel):
        """
        Model representing an intent with a label and a reason.
        """

        intent: Labels
        reason: str

    class IntentPrediction(BaseModel):
        """
        Model representing a list of intents.
        """

        intents: List[Intent]

    cleaned_text = extract_json_from_markdown(response_text)
    parsed_response = json.loads(cleaned_text)
    validated_response = IntentPrediction(**parsed_response)
    response_dict = validated_response.model_dump()

    response_dict["intents"] = [
        {"intent": intent.intent.value, "reason": intent.reason}
        for intent in validated_response.intents
    ]

    return response_dict


@weave.op()
async def call_cohere_with_retry(
    co_client: cohere.AsyncClientV2,
    messages: List[Dict[str, Any]],
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Call the Cohere API with retry logic.

    Args:
        co_client (cohere.AsyncClientV2): The Cohere client to use for the API call.
        messages (List[Dict[str, Any]]): The messages to send to the Cohere API.
        max_retries (int, optional): The maximum number of retries. Defaults to 5.

    Returns:
        Dict[str, Any]: The parsed and validated response from the Cohere API.

    Raises:
        Exception: If the maximum number of retries is reached without successful validation.
    """
    for attempt in range(max_retries):
        response_text = ""
        try:
            response_text = await make_cohere_api_call(
                co_client,
                messages,
                model="command-r-plus",
                temperature=0.0,
                max_tokens=1000,
            )

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
    """A class for enhancing user queries using the Cohere API."""

    @weave.op()
    async def generate_cohere_queries(self, query: str) -> List[str]:
        """
        Generate search queries using the Cohere API.

        Args:
            query (str): The input query for which to generate search queries.

        Returns:
            List[str]: A list of generated search queries.
        """
        co_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))
        # load system prompt
        messages = json.load(open("prompts/search_query.json", "r"))
        # add user prompt (question)
        messages.append({"role": "user", "content": f"## Question\n{query}"})

        response = await co_client.chat(
            model="command-r-plus",
            temperature=0.5,
            max_tokens=500,
            messages=messages,
        )
        search_queries = response.message.content[0].text.splitlines()
        return list(filter(lambda x: x.strip(), search_queries))

    @weave.op()
    async def get_intent_prediction(
        self,
        question: str,
        prompt_file: str = "prompts/intent_prompt.json",
    ) -> Dict[str, Any]:
        """
        Get intent prediction for a given question using the Cohere API.

        Args:
            question (str): The question for which to get the intent prediction.
            prompt_file (str, optional): The file path to the prompt JSON. Defaults to "prompts/intent_prompt.json".

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the intent predictions.
        """
        co_client = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])
        messages = json.load(open(prompt_file))
        messages.append(
            {"role": "user", "content": f"<question>\n{question}\n</question>\n"}
        )

        return await call_cohere_with_retry(co_client, messages)

    @weave.op()
    async def predict(self, query: str) -> Dict[str, Any]:
        """
        Predict the language, generate search queries, and get intent predictions for a given query.

        Args:
            query (str): The input query to process.

        Returns:
            Dict[str, Any]: A dictionary containing the original query, detected language, generated search queries, and intent predictions.
        """
        language = detect_language(query.replace("\n", " "))["lang"]
        search_queries = await self.generate_cohere_queries(query)
        intents = await self.get_intent_prediction(query)
        return {
            "query": query,
            "language": language,
            "search_queries": search_queries,
            "intents": intents["intents"],
        }
