import logging
import os

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

logger = logging.getLogger(__name__)


def load_hyde_prompt(f_name: str = None):
    if f_name and os.path.isfile(f_name):
        hyde_template = open(f_name).read()
    else:
        logger.warning(
            f"No hyde_prompt provided. Using default hyde prompt from {__name__}"
        )
        hyde_template = (
            """Please answer the user's question about the Weights & Biases, W&B or wandb. """
            """Provide a detailed code example with explanations whenever possible. """
            """If the question is not related to Weights & Biases or wandb """
            """just say "I'm not sure how to respond to that."\n"""
            """\n"""
            """Begin\n"""
            """==========\n"""
            """Question: {question}\n"""
            """Answer:"""
        )
    messages = [
        SystemMessagePromptTemplate.from_template(hyde_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    hyde_prompt = ChatPromptTemplate.from_messages(messages)
    return hyde_prompt