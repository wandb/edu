import inspect
import os
import re
from functools import partial
from typing import Dict, List

import cohere
import requests
import weave
from dotenv import load_dotenv
from rich.console import Console
from rich.syntax import Syntax

load_dotenv()


def display_source(symbol):
    # Get the source code
    try:
        source = inspect.getsource(symbol)
    except TypeError:
        print(
            f"Unable to get source code for {symbol}. It might be a built-in or compiled object."
        )
        return

    # Create a Syntax object with the source code
    syntax = Syntax(source, "python", theme="monokai", line_numbers=True)

    # Create a console and print the syntax-highlighted code
    console = Console()
    console.print(syntax)


@weave.op()
def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


@weave.op()
async def make_cohere_api_call(
    co_client: cohere.AsyncClient,
    preamble: str,
    chat_history: List[Dict[str, str]],
    message: str,
    **kwargs,
) -> str:
    response = await co_client.chat(
        preamble=preamble,
        chat_history=chat_history,
        message=message,
        **kwargs,
    )
    return response.text


TOKENIZERS = {
    "command-r": "https://storage.googleapis.com/cohere-public/tokenizers/command-r.json",
    "command-r-plus": "https://storage.googleapis.com/cohere-public/tokenizers/command-r-plus.json",
}


def get_special_tokens_set(tokenizer_url=TOKENIZERS["command-r"]):
    # https://docs.cohere.com/docs/tokens-and-tokenizers
    response = requests.get(tokenizer_url)
    return set([tok["content"] for tok in response.json()["added_tokens"]])


def tokenize_text(text: str, model: str = "command-r") -> List[str]:
    co = cohere.Client(api_key=os.environ["CO_API_KEY"])
    return co.tokenize(text=text, model=model, offline=True)


def length_function(text, model="command-r"):
    return len(tokenize_text(text, model=model).tokens)


length_function_command_r = partial(length_function, model="command-r")
length_function_command_r_plus = partial(length_function, model="command-r-plus")
