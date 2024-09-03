# Erudite: MultiPDF ChatBot 🤖📚

Erudite is an intelligent chatbot that allows you to have conversations with multiple PDF documents.

## Screenshots

<p align="center">
  <img src="Screenshot (11).png" width="1000">
</p>


(Removed this suggested questions code due to some bugs)
<p align="center">
  <img src="Screenshot (12).png" width="1000">
</p>


## 🛠️ How It Works

1. **Document Upload**: Users can upload multiple PDF documents through the Streamlit interface.

2. **Text Extraction**: The `get_pdf_text()` function extracts text from the uploaded PDFs using PyPDF2.

3. **Text Chunking**: The extracted text is split into smaller chunks using `get_text_chunks()` to make it easier to process and analyze.

4. **Vectorization**: The `get_vectorstore()` function creates a vector representation of the text chunks using OpenAI embeddings and stores them in a FAISS index for efficient retrieval.

5. **Conversation Chain**: The `get_conversation_chain()` function sets up a conversational retrieval chain using LangChain, which includes:
   - A custom prompt template
   - The ChatOpenAI language model
   - A conversation memory buffer
   - The FAISS vector store for document retrieval

6. **User Interaction**: Users can ask questions about the uploaded documents through the Streamlit interface.

7. **Answer Generation**: The `handle_userinput()` function processes user questions, retrieves relevant information from the vector store, and generates responses using the conversation chain.

8. **Source Attribution**: The chatbot includes the sources of information in its responses, allowing users to trace the origin of the answers.

## 🌟 Features

- Upload multiple PDF documents 📄
- Process and analyze document content 🔍
- Answer questions based on the uploaded documents 💬
- Provide sources for the information in responses 📚
- User-friendly Streamlit interface 🖥️

## 🧠 Technologies Used

- Streamlit: For the user interface
- LangChain: For building the conversational AI pipeline
- OpenAI: For generating embeddings and powering the language model
- FAISS: For efficient similarity search and clustering of dense vectors
- PyPDF2: For extracting text from PDF documents
