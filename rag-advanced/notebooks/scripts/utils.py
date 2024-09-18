"""
Utility functions for the RAG notebooks.
"""
import inspect
import os
import re
from functools import partial
from typing import Dict, List

import cohere
import requests
import weave
from rich.console import Console
from rich.syntax import Syntax


def display_source(symbol):
    """
    Display the source code of a given symbol using rich's syntax highlighting.

    Args:
        symbol: The symbol (function, class, etc.) whose source code is to be displayed.

    Returns:
        None
    """
    try:
        source = inspect.getsource(symbol)
    except TypeError:
        print(
            f"Unable to get source code for {symbol}. It might be a built-in or compiled object."
        )
        return

    syntax = Syntax(source, "python", theme="monokai", line_numbers=True)

    console = Console()
    console.print(syntax)


@weave.op
def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON from markdown code blocks.

    Args:
        text (str): The markdown text containing JSON code blocks.

    Returns:
        str: The extracted JSON string if found, otherwise the original text.
    """
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


@weave.op
async def make_cohere_api_call(
    co_client: cohere.AsyncClientV2,
    messages: List[Dict[str, any]],
    **kwargs,
) -> str:
    """
    Make an asynchronous API call to the Cohere chat endpoint.

    Args:
        co_client (cohere.AsyncClientV2): The Cohere asynchronous client.
        messages (List[Dict[str, any]]): A list of message dictionaries to send to the chat endpoint.
        **kwargs: Additional keyword arguments to pass to the chat endpoint.

    Returns:
        str: The content of the first message in the response.
    """
    response = await co_client.chat(
        messages=messages,
        **kwargs,
    )
    return response.message.content[0].text


TOKENIZERS = {
    "command-r": "https://storage.googleapis.com/cohere-public/tokenizers/command-r.json",
    "command-r-plus": "https://storage.googleapis.com/cohere-public/tokenizers/command-r-plus.json",
}


def get_special_tokens_set(tokenizer_url=TOKENIZERS["command-r"]):
    """
    Fetches the special tokens set from the given tokenizer URL.

    Args:
        tokenizer_url (str): The URL to fetch the tokenizer from.

    Returns:
        set: A set of special tokens.
    """
    # https://docs.cohere.com/docs/tokens-and-tokenizers
    response = requests.get(tokenizer_url)
    return set([tok["content"] for tok in response.json()["added_tokens"]])


def tokenize_text(text: str, model: str = "command-r") -> List[str]:
    """
    Tokenizes the given text using the specified model.

    Args:
        text (str): The text to be tokenized.
        model (str): The model to use for tokenization. Defaults to "command-r".

    Returns:
        List[str]: A list of tokens.
    """
    co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
    return co.tokenize(text=text, model=model, offline=True)


def length_function(text, model="command-r"):
    """
    Calculate the length of the tokenized text using the specified model.

    Args:
        text (str): The text to be tokenized and measured.
        model (str): The model to use for tokenization. Defaults to "command-r".

    Returns:
        int: The number of tokens in the tokenized text.
    """
    return len(tokenize_text(text, model=model).tokens)


length_function_command_r = partial(length_function, model="command-r")
length_function_command_r_plus = partial(length_function, model="command-r-plus")
