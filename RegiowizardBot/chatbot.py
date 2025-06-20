import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import re
import warnings
import logging
import streamlit as st
from langdetect import detect

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# 🔐 Embed your API key directly for Streamlit Cloud deployment
OPENAI_API_KEY = "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"  # Replace with your actual key

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="RegioWizard KI", layout="centered")
st.title('🧠 RegioWizard KI')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "greetings", "hallo", "servus", "moin"]

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

@st.cache_resource
def get_vectorstore():
    pdf_path = "bad_breisig_docs.pdf"  # Ensure this path is correct on Streamlit Cloud
    loaders = [PyPDFLoader(pdf_path)]
    return VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=300, separators=["\n\n", "\n", ".", "•"]
        )
    ).from_loaders(loaders).vectorstore

def extract_political_groups(text):
    pattern = re.compile(r'(AsF|CDU|SPD|FDP|Junge Union|Senioren-Union|Freie W[aä]hlergruppe)[^\n]*', re.IGNORECASE)
    return '\n'.join(sorted(set([m.group(0).strip() for m in pattern.finditer(text)])))

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        openai_chat = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )

        lang = detect_language(prompt)

        if is_greeting(prompt):
            response = "Hallo, ich bin der RegioWizard_KI Chatbot! 😊 Frag mich alles über Bad Breisig!" if lang == "de" else "Hi, I'm RegioWizard_KI Chatbot! 😊 Ask me anything about Bad Breisig!"
        else:
            vectorstore = get_vectorstore()

            qa_prompt = ChatPromptTemplate.from_template("""
{prefix}

Context:
{context}

{q_prefix}: {question}

{a_prefix}:
""").partial(
                prefix="Du bist ein hilfsbereiter Assistent mit Wissen über Bad Breisig. Verwende AUSSCHLIESSLICH den untenstehenden Kontext, um die Frage des Nutzers zu beantworten." if lang == "de" else "You are a helpful assistant knowledgeable about Bad Breisig. Use ONLY the context below to answer the user's question.",
                q_prefix="Frage" if lang == "de" else "Question",
                a_prefix="Antwort" if lang == "de" else "Answer"
            )

            chain = RetrievalQA.from_chain_type(
                llm=openai_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 12}),
                chain_type_kwargs={"prompt": qa_prompt},
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"].strip()

            if any(x in prompt.lower() for x in ["partei", "gruppierung", "gruppen", "parties", "political"]):
                fallback_docs = result.get("source_documents", [])
                combined_text = "\n".join(doc.page_content for doc in fallback_docs)
                filtered = extract_political_groups(combined_text)
                if filtered:
                    response = f"Die politischen Gruppierungen in Bad Breisig sind:\n\n{filtered}" if lang == "de" else f"The political groups in Bad Breisig are:\n\n{filtered}"

            if not response or "not found" in response.lower() or "nicht im kontext" in response.lower():
                fallback_docs = vectorstore.similarity_search_with_score(prompt, k=3)
                keyword_hits = list({doc.page_content.strip()[:300] for doc, _ in fallback_docs})

                if keyword_hits:
                    response = "Hier sind die relevantesten Informationen:\n\n" if lang == "de" else "Here’s the most relevant information found:\n\n"
                    response += "\n\n".join(keyword_hits)
                else:
                    response = "Nicht im bereitgestellten Dokument gefunden." if lang == "de" else "Not found in the provided document."

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
