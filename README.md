# Project: LangChain Integration with Google Generative AI

## Overview

This project demonstrates the integration of LangChain with Google Generative AI to provide a powerful question-answering system using document retrieval and embeddings. The goal is to process and analyze documents using machine learning models, and retrieve relevant information based on queries.

---

## Setup

### Prerequisites

Before starting, make sure you have the following libraries installed:

```bash
!pip -q install langchain huggingface_hub openai tiktoken pypdf
!pip -q install google-generativeai faiss-cpu chromadb unstructured
!pip -q install sentence_transformers
!pip -q install -U FlagEmbedding
```

### Environment Configuration

Set the environment variables for LangChain:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key_here"
os.environ["LANGCHAIN_PROJECT"] = "your_project_here"
```

Make sure to enter your Google AI API key if it’s not already configured:

```python
import getpass
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
```

### Mounting Google Drive

To work with documents stored in Google Drive, mount the drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Document Loading & Embeddings

### Loading Documents

Load PDF documents using LangChain’s `PyPDFLoader`:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/drive/MyDrive/your_file.pdf")
docs = loader.load()
```

### Text Splitting

The text from the documents is split into smaller chunks for efficient retrieval using `RecursiveCharacterTextSplitter`:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
```

### Embeddings

The BGE embeddings are used for indexing and similarity search:

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
```

---

## Retrieval & Question Answering

### Vector Store and Retriever

Create a `Chroma` vector store and a parent document retriever to fetch relevant documents based on queries:

```python
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever

vectorstore = Chroma(collection_name="split_parents", embedding_function=hf)
store = InMemoryStore()
retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter)
```

### Query Example

Run a query and retrieve relevant documents:

```python
sub_docs = vectorstore.similarity_search("what is stock market")
print(sub_docs[0].page_content)
```

### Using Google Generative AI for Question Answering

Install the Google Generative AI package:

```bash
!pip install --upgrade --quiet langchain-google-genai
```

Now, create an instance of the `ChatGoogleGenerativeAI`:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2)
```

### Run QA

Use `RetrievalQA` to combine the retriever and model to answer queries:

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
query = "what is stock market"
qa.run(query)
```

---

## Conclusion

This project demonstrates how to integrate LangChain with Google Generative AI for an advanced document retrieval and question-answering system. It uses embeddings for document indexing and retrieval, allowing for efficient and relevant results based on user queries.
