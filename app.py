from dotenv import load_dotenv
from utils.loaders import load_documents
from utils.loaders import load_documents, clean_metadata
from utils.embeddings import get_embeddings
from utils.vectordb import create_vector_store
from utils.splitter import split_documents

import os
import shutil

# Load environment variables
load_dotenv()

# Read API key
api_key = os.getenv("GROQ_API_KEY")

print("API Key loaded:", api_key[:10], "...")

# Startup cleanup function
def startup_cleanup():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("Old Chroma DB deleted")
    else:
        print("No previous DB found")

# Run cleanup
startup_cleanup()

documents = load_documents()
print("Loaded documents:", len(documents))

documents = clean_metadata(documents)
print("Metadata cleaned")

chunks = split_documents(documents)
print("Number of chunks:", len(chunks))

embeddings = get_embeddings()
print("Embedding model loaded")

vectordb = create_vector_store(chunks, embeddings)
print("Vector database created successfully")


