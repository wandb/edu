# SOURCE: https://github.com/hwchase17/langchain-streamlit-template/blob/master/main.py

import streamlit as st
from streamlit_chat import message

from wandb.integration.langchain import WandbTracer

from config import default_config, wandb_config
from chain import load_chain

chain = load_chain(default_config)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "How do I share a W&B report with team members?", key="input")
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