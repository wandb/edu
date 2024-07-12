import json
import os

import cohere
import nest_asyncio
import weave
from dotenv import load_dotenv
from ftlangdetect import detect as detect_language
from syncer import sync

nest_asyncio.apply()
import asyncio
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from .utils import extract_json_from_markdown, make_cohere_api_call

load_dotenv()


@weave.op()
async def parse_and_validate_response(response_text: str) -> Dict[str, Any]:
    """Parse and validate the response text."""

    class Labels(str, Enum):
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
        intent: Labels
        reason: str

    class IntentPrediction(BaseModel):
        intents: List[Intent]

        @field_validator("intents")
        def check_unique_intents(cls, v):
            intent_set = set(intent.intent for intent in v)
            if len(intent_set) != len(v):
                raise ValueError("Intents must be unique")
            return v

    cleaned_text = extract_json_from_markdown(response_text)
    parsed_response = json.loads(cleaned_text)
    validated_response = IntentPrediction(**parsed_response)
    # Convert the validated response to a dictionary
    response_dict = validated_response.model_dump()

    # Replace enum keys with their values
    response_dict["intents"] = [
        {"intent": intent.intent.value, "reason": intent.reason}  # Use the enum value
        for intent in validated_response.intents
    ]

    return response_dict


@weave.op()
async def call_cohere_with_retry(
    co_client: cohere.AsyncClient,
    preamble: str,
    chat_history: List[Dict[str, str]],
    message: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        response_text = ""
        try:
            response_text = await make_cohere_api_call(
                co_client,
                preamble,
                chat_history,
                message,
                model="command-r-plus",
                force_single_step=True,
                temperature=0.0,
                prompt_truncation="AUTO",
                max_tokens=1000,
            )

            return await parse_and_validate_response(response_text)
        except Exception as e:
            error_message = f"Your previous response resulted in an error: {str(e)}"

            if attempt == max_retries - 1:
                raise

            chat_history.extend(
                [
                    {"role": "USER", "message": message},
                    {"role": "CHATBOT", "message": response_text},
                ]
            )
            message = f"{error_message}\nPlease provide a valid JSON response based on the previous context and error message. Ensure that:\n1. The response is a valid JSON object with an 'intents' list.\n2. Each intent in the list has a valid 'intent' from the Labels enum and a 'reason'.\n3. The intents are unique.\n4. The response is not wrapped in markdown code blocks."

    raise Exception("Max retries reached without successful validation")


class QueryEnhancer(weave.Model):
    @weave.op()
    async def generate_cohere_queries(self, query: str) -> List[str]:
        co_client = cohere.AsyncClient(api_key=os.getenv("CO_API_KEY"))
        search_gen_prompt = json.load(open("prompts/search_query.json", "r"))

        response = await co_client.chat(
            model="command-r-plus",
            preamble=search_gen_prompt[0]["message"],
            message=f"## Question\n{query}",
            temperature=0.5,
            # search_queries_only=True,
            max_tokens=500,
        )
        return response.text.splitlines()

    @weave.op()
    async def get_intent_prediction(
        self,
        question: str,
        prompt_file: str = "prompts/intent_prompt.json",
    ) -> List[Dict[str, Any]]:
        co_client = cohere.AsyncClient(api_key=os.environ["CO_API_KEY"])
        messages = json.load(open(prompt_file))
        preamble = messages[0]["message"]
        chat_history = []
        message_template = """<question>\n{question}\n</question>\n"""
        message = message_template.format(question=question)

        return await call_cohere_with_retry(co_client, preamble, chat_history, message)

    @weave.op()
    async def __call__(self, query: str) -> str:
        language = detect_language(query)["lang"]
        search_queries = await self.generate_cohere_queries(query)
        intents = await self.get_intent_prediction(query)
        return {
            "query": query,
            "language": language,
            "search_queries": search_queries,
            "intents": intents["intents"],
        }

    @weave.op()
    async def predict(self, query: str) -> str:
        return await self.__call__(query)
