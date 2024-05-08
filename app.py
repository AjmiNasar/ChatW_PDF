import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import openai
from langchain.chains.question_answering import load_qa_chain
import pickle 
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
import os

# sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by [ajmism]')

def main():
    st.header("Chat with PDF💬")
    load_dotenv()
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.write(pdf.name)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pk1"):
         with open(f"{store_name}.pkl", "rb") as f:
            vectorstore =pickle.load(f)
        else:
            embeddings=OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
             pickle.dump(vectorstore, f)
    #accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")
    if query:
        docs=vectorstore.similarity_search(query=query,k=3)
        llm=openai(model_name='gpt-3.5-turbo')
        chain=load_qa_chain(llm=llm,chain_type="stuff")
        with get_openai_callback() as cb:
           response=chain.run(input_documents=docs,question=query)
           print(cb)
        st.write(response)
if __name__ == '__main__':
    main()
