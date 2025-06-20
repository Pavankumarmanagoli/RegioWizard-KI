import warnings
import logging
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load OpenAI key
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("AW_Wiki_ChatBOT")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('enter a message')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    sys_prompt = ChatPromptTemplate.from_template("""
You are a very good AI assistant named "WIKI-CHAT". You always provide precise and concise answers. Be nice and polite.
""")

    openai_chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    chain = sys_prompt | openai_chat | StrOutputParser()
    response = chain.invoke({"user_prompt": prompt})

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
