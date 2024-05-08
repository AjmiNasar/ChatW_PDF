import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import openai
from langchain.chains.question_answering import load_qa_chain
import pickle 
from dotenv import load_dotenv
import os

# Function to process PDF and return text chunks
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

# Function to load or create vector store
def load_or_create_vectorstore(pdf_name, chunks):
    store_name = os.path.splitext(pdf_name)[0]
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
    return vectorstore

def main():
    st.header("Chat with PDF💬")
    load_dotenv()
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        st.write(pdf.name)
        try:
            chunks = process_pdf(pdf)
            vectorstore = load_or_create_vectorstore(pdf.name, chunks)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return

        query = st.text_input("Ask questions about your PDF file:")
        if query:
            try:
                docs = vectorstore.similarity_search(query=query, k=3)
                llm = openai(model_name='gpt-3.5-turbo')
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with st.spinner("Searching for answers..."):
                    response = chain.run(input_documents=docs, question=query)
                st.write(response)
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == '__main__':
    main()
