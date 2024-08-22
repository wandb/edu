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

        # TODO: do we really need this validation?
        # given I don't see this grounding in the system prompt!
        # @field_validator("intents")
        # def check_unique_intents(cls, v):
        #     intent_set = set(intent.intent for intent in v)
        #     if len(intent_set) != len(v):
        #         raise ValueError("Intents must be unique")
        #     return v

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
    co_client: cohere.AsyncClientV2,
    messages: List[Dict[str, Any]],
    max_retries: int = 5,
) -> Dict[str, Any]:
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
            error_message = f"{error_message}\nPlease provide a valid JSON response based on the previous context and error message. Ensure that:\n1. The response is a valid JSON object with an 'intents' list.\n2. Each intent in the list has a valid 'intent' from the Labels enum and a 'reason'.\n3. The intents are unique.\n4. The response is not wrapped in markdown code blocks."

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
    @weave.op()
    async def generate_cohere_queries(self, query: str) -> List[str]:
        co_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))
        # load system prompt
        messages = json.load(open("prompts/search_query.json", "r"))
        # add user prompt (question)
        messages.append(
            {"role": "user", "content": f"## Question\n{query}"}
        )

        response = await co_client.chat(
            model="command-r-plus",
            temperature=0.5,
            max_tokens=500,
            messages=messages,
        )
        return response.message.content[0].text.splitlines()

    @weave.op()
    async def get_intent_prediction(
        self,
        question: str,
        prompt_file: str = "prompts/intent_prompt.json",
    ) -> List[Dict[str, Any]]:
        co_client = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])
        messages = json.load(open(prompt_file))
        messages.append({"role": "user", "content": f"<question>\n{question}\n</question>\n"})

        return await call_cohere_with_retry(co_client, messages)

    @weave.op()
    async def __call__(self, query: str) -> str:
        language = detect_language(query.replace('\n', ' '))["lang"]
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
