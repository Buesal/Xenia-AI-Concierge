import streamlit as st
import os
import tempfile
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


st.set_page_config(page_title="Xenia AI | Hotel Concierge", page_icon="🛎️", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;} 
footer {visibility: hidden;} 
header {visibility: hidden;}
.stChatInputContainer {padding-bottom: 20px;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo Xenia AI.png", use_container_width=True)

st.markdown("<h4 style='text-align: center; color: #555;'>Upload the Hotel Directory (PDF) to activate the multilingual AI concierge.</h4>", unsafe_allow_html=True)
st.write("")

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Error: OpenAI API key not found. Please check your .env file!")
    st.stop()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.image("logo Xenia AI.png", use_container_width=True)
    st.header("⚙️ Hotel Admin Panel")
    uploaded_file = st.file_uploader("Upload Hotel Directory (PDF)", type="pdf")

    if uploaded_file and st.button("Activate AI Concierge"):
        with st.spinner("Processing hotel rules..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            unique_collection_name = f"pdf_{uuid.uuid4().hex}"
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=unique_collection_name
            )

            st.session_state.vector_store = vector_store

            st.session_state.messages = [
                {"role": "assistant",
                 "content": "Welcome! I am Xenia, your AI Concierge. How can I make your stay more comfortable today?"}
            ]

            os.remove(temp_file_path)
            st.success("Xenia AI is online and ready to assist guests!")


for message in st.session_state.messages:
    avatar_icon = "🛎️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask about breakfast, spa, or hotel rules..."):

    if st.session_state.vector_store is None:
        st.warning("Please activate the AI Concierge in the admin panel first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🛎️"):
            with st.spinner("Looking up hotel policies..."):
                llm = ChatOpenAI(model="gpt-4.1", temperature=0)

                system_prompt = (
                    "You are Xenia, a polite, premium AI concierge for a luxury hotel in Greece.\n"
                    "Your task is to answer guest questions using ONLY the provided context.\n\n"
                    "RULES:\n"
                    "1. Multilingualism: Always answer in the language of the user's question.\n"
                    "2. Tone: Be exceptionally polite, professional, and welcoming.\n"
                    "3. Accuracy: Use ONLY information from the context.\n"
                    "4. Honesty: If the answer is not in the context, say exactly: 'I apologize, but I do not have that information at the moment. Please contact the front desk.'\n"
                    "5. Format: Provide the answer clearly.\n\n"
                    "RESPONSE FORMAT:\n"
                    "**Answer:** <your accurate and polite answer>\n"
                    "**Policy Reference:** <a brief 1-sentence summary from the context>\n"
                    "**Confidence:** <High / Medium / Low>\n\n"
                    "Context:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke({"input": user_question})
                answer_text = response["answer"]

                st.markdown(answer_text)

                with st.expander("🔍 View Official Policy Fragment"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Fragment {i + 1}:**\n{doc.page_content}")
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer_text})