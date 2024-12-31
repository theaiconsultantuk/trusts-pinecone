# pinecone_upsert_pdfs_query.py

import os
import pdfplumber
import openai
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# ----------------------------
# Configuration from .env
# ----------------------------

# OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# Pinecone Configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# MongoDB Credentials (if needed elsewhere in your project)
mongodb_username = os.getenv("MONGODB_USERNAME")
mongodb_password = os.getenv("MONGODB_PASSWORD")
mongodb_host = os.getenv("MONGODB_HOST")
mongodb_db = os.getenv("MONGODB_DB")

# PDF Directory
pdf_directory = os.getenv("PDF_DIRECTORY")

# ----------------------------
# Validation of Environment Variables
# ----------------------------

# Validate OpenAI Configuration
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
openai.api_key = openai_api_key

# Validate Pinecone Configuration
if not pinecone_api_key or not pinecone_env:
    raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment variables.")

# Validate PDF Directory
if not pdf_directory:
    raise ValueError("PDF_DIRECTORY not set in environment variables.")
if not os.path.isdir(pdf_directory):
    raise ValueError(f"PDF_DIRECTORY path '{pdf_directory}' does not exist or is not a directory.")

# ----------------------------
# Initialize Pinecone
# ----------------------------

# Create an instance of the Pinecone client
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Access the existing Pinecone index
index_name = "trusts-index"

# Verify if the index exists
available_indexes = pc.list_indexes()
if index_name not in available_indexes:
    raise ValueError(f"Index '{index_name}' does not exist in your Pinecone account. Available indexes: {available_indexes}")

# Initialize the index
pinecone_index = pc.Index(index_name)

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
        raise RuntimeError(f"Error extracting text from {pdf_path}: {e}")

def embed_text(text):
    """
    Embeds the given text using OpenAI's embedding model.
    """
    try:
        response = openai.Embedding.create(
            model=openai_model,
            input=text
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        raise RuntimeError(f"Error embedding text: {e}")

# ----------------------------
# Upsert PDF Embeddings to Pinecone
# ----------------------------

def upsert_pdfs():
    """
    Processes each PDF in the specified directory, extracts text,
    embeds it, and upserts the embedding into Pinecone with metadata.
    """
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            
            try:
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(pdf_path)
                
                if not pdf_text.strip():
                    print(f"No text found in {filename}. Skipping.")
                    continue
                
                # Get embedding for extracted text
                embedding = embed_text(pdf_text)
                
                # Prepare and upsert embedding with metadata
                metadata = {"filename": filename, "text": pdf_text}
                pinecone_index.upsert(
                    vectors=[
                        {
                            "id": filename,  # Unique identifier for the vector
                            "values": embedding,
                            "metadata": metadata
                        }
                    ]
                )
                
                print(f"Upserted embedding for {filename}")
            
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# ----------------------------
# Query Pinecone Index
# ----------------------------

def query_pinecone(query_text, top_k=3):
    """
    Queries the Pinecone index with the provided text and returns the top_k matches.
    """
    try:
        # Embed the query text
        query_embedding = embed_text(query_text)
        
        # Perform the query
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        # Display query results
        print("\nQuery Results:")
        for match in results['matches']:
            print(f"ID: {match['id']}, Metadata: {match['metadata']}")
    
    except Exception as e:
        print(f"Query failed: {e}")

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    # Upsert PDFs
    upsert_pdfs()
    
    # Perform a sample query
    sample_query = "What are the two types of Trust?"
    query_pinecone(sample_query)
