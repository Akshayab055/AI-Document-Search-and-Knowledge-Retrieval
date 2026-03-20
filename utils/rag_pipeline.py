from utils.loaders import load_documents, clean_metadata
from utils.splitter import split_documents
from utils.embeddings import get_embeddings
from utils.vectordb import create_vector_store
from utils.retriever import get_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM once
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Build RAG system once
def initialize_rag():
    documents = load_documents()
    documents = clean_metadata(documents)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectordb = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vectordb)
    return retriever

retriever = initialize_rag()

# Function to get answer
def get_answer(query, chat_history):
    # Combine previous conversation
    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.

        Use the conversation history and context to answer.

        If answer not found, say "Not available in documents."

        Conversation History:
        {history}

        Context:
        {context}

        Question:
        {question}
        """
    )

    formatted_prompt = prompt.format(
        history=history_text,
        context=context,
        question=query
    )

    response = llm.invoke(formatted_prompt)

    sources = list(set([doc.metadata.get("source") for doc in retrieved_docs]))

    return response.content, sources