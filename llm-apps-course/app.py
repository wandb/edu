# SOURCE: https://github.com/hwchase17/langchain-streamlit-template/blob/master/main.py

import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from wandb.integration.langchain import WandbTracer
import wandb

from config import default_config, wandb_config
from ingest import DocumentStore

def load_artifacts(config):
    vector_store_artifact = wandb.use_artifact(config.vector_store_artifact, type="search_index")
    vector_store_artifact_dir = vector_store_artifact.download()
    return vector_store_artifact_dir

def load_vector_store(store_dir):
    document_store = DocumentStore.load_from_disk(store_dir)
    return document_store

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    vector_store_artifact_dir = load_artifacts(default_config)
    vector_store = load_vector_store(vector_store_artifact_dir)
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa

wandb.init(project="llmapps")

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "How can I share my W&B report with team members?", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = chain.run(query=user_input, callbacks=[WandbTracer(wandb_config)])

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")