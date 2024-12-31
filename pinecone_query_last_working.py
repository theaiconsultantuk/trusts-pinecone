# pinecone_query.py

import os
import streamlit as st
from openai import OpenAI
import pinecone
from pinecone import Pinecone
import logging

# ----------------------------
# Logging Configuration
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Load Environment Variables
# ----------------------------

# Access secrets using Streamlit's st.secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_model = st.secrets.get("OPENAI_MODEL", "text-embedding-3-small")
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]

# Pinecone Index Configuration
index_name = "trusts-index"
namespace = "ns1"  # Ensure this matches the namespace used during upsert

# ----------------------------
# Validate Environment Variables
# ----------------------------

if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in Streamlit secrets.")
    raise ValueError("OPENAI_API_KEY not found in Streamlit secrets.")

if not pinecone_api_key or not pinecone_env:
    logger.error("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in Streamlit secrets.")
    raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in Streamlit secrets.")

# ----------------------------
# Initialize OpenAI and Pinecone Clients
# ----------------------------

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    logger.info("Pinecone initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# Access the Pinecone index
try:
    pinecone_index = pc.Index(index_name)
    logger.info(f"Pinecone index '{index_name}' accessed successfully.")
except Exception as e:
    logger.error(f"Failed to access Pinecone index '{index_name}': {e}")
    raise

# ----------------------------
# Function Definitions
# ----------------------------

def embed_query(query_text):
    """
    Embeds the query text using OpenAI's embedding model.
    """
    try:
        response = client.embeddings.create(
            model=openai_model,
            input=query_text
        )
        embedding = response.data[0].embedding  # Corrected access
        return embedding
    except Exception as e:
        logger.error(f"Error embedding query text: {e}")
        return None

def query_pinecone(embedding, top_k=5):
    """
    Queries the Pinecone index with the given embedding and returns top_k matches.
    """
    try:
        response = pinecone_index.query(
            vector=embedding,
            top_k=top_k,
            include_values=False,       # We don't need the vector values
            include_metadata=True,      # To retrieve metadata like text and filename
            namespace=namespace
        )
        return response
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return None

def display_results(response):
    """
    Displays the query results.
    """
    if not response or not response.matches:
        logger.info("No matches found.")
        return

    logger.info(f"Top {len(response.matches)} results:")
    for match in response.matches:
        metadata = match.metadata
        filename = metadata.get("filename", "Unknown File")
        chunk_id = metadata.get("chunk_id", "Unknown Chunk")
        text = metadata.get("text", "No text available.")
        logger.info(f"Filename: {filename}, Chunk ID: {chunk_id}\nText: {text}\n{'-'*80}")

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    # Define your query
    user_query = "What are the two types of Trust?"

    logger.info(f"Embedding the query: '{user_query}'")
    query_embedding = embed_query(user_query)
    if not query_embedding:
        logger.error("Failed to generate embedding for the query.")
        exit(1)

    logger.info("Querying Pinecone for similar vectors...")
    query_response = query_pinecone(query_embedding, top_k=5)
    if not query_response:
        logger.error("Failed to retrieve query results from Pinecone.")
        exit(1)

    logger.info("Displaying query results:")
    display_results(query_response)
