import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, model_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunk, model_name, api_key=None):
    if model_name == "Ollama":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = FAISS.from_texts(text_chunk, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "Ollama":
        prompt_template = """
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the context, say so and suggest doing a web search.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        model = ChatOllama(model="llama3.2")
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if pdf_docs is None:
        st.warning("Please upload PDF files before processing.")
        return

    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name)

    user_question_output = ""
    response_output = ""

    if model_name == "Ollama":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(model_name, vectorstore=new_db)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

        st.markdown(
            f"""
            <style>
                .chat-message {{ padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; }}
                .chat-message.user {{ background-color: #2b313e; }}
                .chat-message.bot {{ background-color: #475063; }}
                .chat-message .avatar {{ width: 20%; }}
                .chat-message .avatar img {{ max-width: 78px; max-height: 78px; border-radius: 50%; object-fit: cover; }}
                .chat-message .message {{ width: 80%; padding: 0 1.5rem; color: #fff; }}
            </style>
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{user_question_output}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{response_output}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if len(conversation_history) == 1:
        conversation_history = []
    elif len(conversation_history) > 1:
        last_item = conversation_history[-1]
        conversation_history.remove(last_item)

    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    st.snow()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs using Ollama (local)")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("Select the Model:", ("Ollama",))
    api_key = None

    with st.sidebar:
        st.title("Menu:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
            pdf_docs = None
        elif clear_button:
            if 'user_question' in st.session_state:
                st.warning("The previous query will be discarded.")
                st.session_state.user_question = ""
                if len(st.session_state.conversation_history) > 0:
                    st.session_state.conversation_history.pop()

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""

if __name__ == "__main__":
    main()