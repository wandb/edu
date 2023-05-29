from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import wandb

from ingest import DocumentStore

def load_vector_store(config):
    # check if wandb run exist and create one if not
    if wandb.run is None:
        wandb.init(project=config.project)
    # load vector store artifact
    vector_store_artifact_dir = wandb.use_artifact(config.vector_store_artifact, type="search_index").download()
    # load vector store
    document_store = DocumentStore.load_from_disk(vector_store_artifact_dir)
    return document_store

def load_chain(config):
    """Logic for loading the chain you want to use should go here."""
    vector_store = load_vector_store(config)
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa
