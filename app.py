import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
# from embeddings import TfidfEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text_data = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_data.append({"text": text, "source": pdf.name})
    return text_data

def get_text_chunks(text_data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for item in text_data:
        chunks += [{"chunk": chunk, "source": item["source"]} for chunk in text_splitter.split_text(item["text"])]
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    texts = [item["chunk"] for item in text_chunks]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=[{"source": item["source"]} for item in text_chunks])
    return vectorstore

def get_conversation_chain(vectorstore):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant designed to answer questions based on the provided context. Your responses should be accurate and directly related to the information in the context. If the question cannot be answered using the given context, politely inform the user that you don't have enough information to answer and suggest they ask a related question that might be covered in the knowledge base.

        Always include the sources of the information in your response.

        Context: {context}
        Question: {question}

        Answer:
        """
    )

    llm = ChatOpenAI(
        temperature=0,
        max_tokens=500,
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'  
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            bot_response = message.content
            
            if 'source_documents' in response:
                sources = [doc.metadata['source'] for doc in response['source_documents']]
                unique_sources = list(set(sources)) 
                source_string = "\n\nSources: " + ", ".join(unique_sources)
                bot_response += source_string

            st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Buzzword",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Erudite: MultiPDF ChatBot 🤖📚")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text_data = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text_data)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()