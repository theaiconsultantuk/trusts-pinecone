# pinecone_upsert_pdfs.py

import os
import pdfplumber
from openai import OpenAI
import pinecone
from pinecone import Pinecone
import streamlit as st
import logging

# Load environment variables from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_model = st.secrets.get("OPENAI_MODEL", "text-embedding-3-small")
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]

# PDF Directory
pdf_directory = "uploads"

# Ensure the directory exists
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)

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
# Validation of Environment Variables
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

# Access the existing Pinecone index
index_name = "trusts-index"

try:
    pinecone_index = pc.Index(index_name)
    logger.info(f"Pinecone index '{index_name}' accessed successfully.")
except Exception as e:
    logger.error(f"Failed to access Pinecone index '{index_name}': {e}")
    raise

# ----------------------------
# Function Definitions
# ----------------------------

def extract_text_from_pdf(pdf_path):
    """
    Extracts and returns all text from a PDF file.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def split_text(text, max_length=500):
    """
    Splits text into chunks of a specified maximum length.
    """
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

def embed_text(text):
    """
    Embeds the given text using OpenAI's embedding model.
    """
    try:
        response = client.embeddings.create(
            model=openai_model,
            input=text
        )
        embedding = response.data[0].embedding  # Corrected access
        return embedding
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        return None

def embed_text_chunks(text, chunk_size=500):
    """
    Embeds text in smaller chunks and returns a list of embeddings.
    """
    chunks = split_text(text, max_length=chunk_size)
    embeddings = []
    for chunk in chunks:
        embed = embed_text(chunk)
        if embed:
            embeddings.append(embed)
    return embeddings

# ----------------------------
# Embed & Upsert PDFs
# ----------------------------

def embed_and_upsert_pdfs(file_paths):
    """
    Embeds PDFs from given file paths and upserts them into the Pinecone index with metadata.
    """
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        logger.info(f"Processing file: {filename}")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(file_path)
        if not pdf_text.strip():
            logger.warning(f"No text found in {filename}. Skipping.")
            continue
        logger.debug(f"Extracted text from {filename}.")
        
        # Split text into chunks
        text_chunks = list(split_text(pdf_text))
        logger.info(f"Split {filename} into {len(text_chunks)} chunks.")
        
        # Embed each chunk
        embeddings = embed_text_chunks(pdf_text)
        if not embeddings:
            logger.warning(f"No embeddings generated for {filename}. Skipping upsert.")
            continue
        logger.debug(f"Generated embeddings for {filename}.")
        
        # Prepare vectors for upsert
        vectors = []
        for idx, embed in enumerate(embeddings, start=1):
            vector_id = f"{filename}_chunk_{idx}"
            vectors.append({
                "id": vector_id,
                "values": embed,
                "metadata": {
                    "filename": filename,
                    "chunk_id": idx,
                    "text": text_chunks[idx - 1]
                }
            })
        logger.debug(f"Prepared {len(vectors)} vectors for {filename}.")
        
        # Upsert vectors into Pinecone with namespace
        try:
            upsert_response = pinecone_index.upsert(
                vectors=vectors,
                namespace="ns1"  # Specify your desired namespace
            )
            logger.info(f"Upserted {len(vectors)} vectors for {filename} into namespace 'ns1'.")
            logger.debug(f"Upsert response: {upsert_response}")
        except Exception as e:
            logger.error(f"Failed to upsert vectors for {filename}: {e}")

# ----------------------------
# Query Pinecone Index (Commented Out)
# ----------------------------

# def query_pinecone(query_text, top_k=3):
#     """
#     Queries the Pinecone index with the provided text and returns the top_k matches.
#     """
#     try:
#         # Embed the query text
#         query_embedding = embed_text(query_text)
#         if not query_embedding:
#             logger.error("Failed to generate embedding for the query.")
#             return
#         
#         # Perform the query
#         results = pinecone_index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_values=False,
#             include_metadata=True,
#             namespace="ns1"
#         )
#         
#         # Display query results
#         logger.info("Query Results:")
#         for match in results['matches']:
#             print(f"ID: {match['id']}, Metadata: {match['metadata']}")
#     
#     except Exception as e:
#         logger.error(f"Query failed: {e}")

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    # Embed and Upsert PDFs
    file_paths = [os.path.join(pdf_directory, filename) for filename in os.listdir(pdf_directory) if filename.lower().endswith(".pdf")]
    embed_and_upsert_pdfs(file_paths)
    
    # Perform a sample query (Commented Out)
    # sample_query = "What are the two types of Trust?"
    # query_pinecone(sample_query)
