from backend.core import run_llm
import streamlit as st

st.header("Langchain- Documentation Helper Bot")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if ("user_prompt_history" not in st.session_state and  "chat_answer_history" not in st.session_state and  "chat_history" not in st.session_state):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(sources: str)->str:
    if not sources: return ""
    sources_list = list(sources)
    sources_list.sort()
    sources_str = "Sources: \n"

    for i, src in enumerate(sources_list):
        sources_str += f"{i+1}. {src}\n"
    return sources_str

if prompt:
    with st.spinner("Generating Response"):
        generated_response = run_llm(query = prompt, chat_history = st.session_state["chat_history"])

        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt )
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response['result']))

if st.session_state["chat_answer_history"]:
    for gen_res, usr_qry in zip(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(usr_qry)
        st.chat_message("assistant").write(gen_res)