Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload documents and ask questions based on their content. It combines document retrieval with a language model to generate accurate, context-aware responses.

Features
RAG-based question answering using document context
Upload and process PDF and TXT files
Semantic search using embeddings
Conversational chat with history support
Session-based memory (no persistence)
Dynamic document addition without restart
Corrective RAG with relevance checking
Real-time UI updates using state management
Interactive chat interface with navigation
Architecture

User → Upload → Documents → RAG Backend → Vector Store → Retriever → LLM → Response

Tech Stack

Frontend

Reflex (Python-based UI framework)

Backend

Python (RAG pipeline)

AI Framework

LangChain

LLM

Groq / LLaMA

Vector Database

ChromaDB

File Storage

Local /documents folder

How It Works

User uploads documents
Documents are processed into embeddings
Stored in vector database
User asks a question
Retriever fetches relevant chunks
LLM generates response using context and history

Conclusion

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) based chatbot that integrates document retrieval with a language model to provide accurate and context-aware responses.

It showcases key concepts such as embeddings, vector databases, and conversational memory while maintaining a clean and modular architecture.

The system provides an efficient way to query documents in real time and can be further extended for scalable, production-level applications.
