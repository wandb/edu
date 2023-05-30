import logging
from types import SimpleNamespace

import wandb
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

logger = logging.getLogger(__name__)


def load_vector_store(config: SimpleNamespace):
    """Load a vector store from a Weights & Biases artifact
    Args:
        config (SimpleNamespace): A config object that contains the artifact name
    Returns:
        Chroma: A Chroma vector store object
    """
    # check if wandb run exist and create one if not
    if wandb.run is None:
        wandb.init(project=config.project)
    # load vector store artifact
    vector_store_artifact_dir = wandb.use_artifact(
        config.vector_store_artifact, type="search_index"
    ).download()
    embedding_fn = OpenAIEmbeddings()
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )

    return vector_store


def load_chain(config: SimpleNamespace):
    """Load a ConversationalQA chain from a config

    Args:
        config (SimpleNamespace): A config object

    """
    if wandb.run is None:
        wandb.init(project=config.project)
    vector_store = load_vector_store(config)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        model_name=config.model_name,
        temperature=config.chat_temperature,
        max_retries=config.max_fallback_retries,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=retriever, qa_prompt=config.qa_prompt
    )
    return qa
