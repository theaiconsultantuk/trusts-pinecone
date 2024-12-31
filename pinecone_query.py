import os
from dotenv import load_dotenv
from openai import OpenAI
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

load_dotenv()

# OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "text-embedding-ada-002")

# Pinecone Configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# Pinecone Index Configuration
index_name = "trusts-index"
namespace = "ns1"

# ----------------------------
# Validate Environment Variables
# ----------------------------

if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

if not pinecone_api_key or not pinecone_env:
    logger.error("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment variables.")
    raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment variables.")

# ----------------------------
# Initialize OpenAI and Pinecone Clients
# ----------------------------

# Initialize OpenAI client
try:
    client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=pinecone_api_key)
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
            input=query_text,
            model=openai_model
        )
        embedding = response.data[0].embedding
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
            include_values=False,
            include_metadata=True,
            namespace=namespace
        )
        return response
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return None

def display_results(response):
    """
    Displays the query results with similarity scores.
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
        similarity_score = match.score
        percentage_match = similarity_score * 100
        logger.info(f"Filename: {filename}, Chunk ID: {chunk_id}, Similarity: {percentage_match:.2f}%")
        logger.info(f"Text: {text}\n{'-'*80}")

def prepare_ai_prompt(response, question):
    """
    Prepares the prompt for the AI using the query results.
    """
    if not response or not response.matches:
        return None

    prompt = (
        "You are an AI assistant. Based on the following excerpts from various documents, "
        "answer the question below as thoroughly as possible. If the provided data is insufficient, "
        "state that there is insufficient relevant data.\n\n"
        f"Question: {question}\n\n"
        "Excerpts:\n"
    )

    for match in response.matches:
        metadata = match.metadata
        filename = metadata.get("filename", "Unknown File")
        chunk_id = metadata.get("chunk_id", "Unknown Chunk")
        text = metadata.get("text", "No text available.")
        similarity_score = match.score
        percentage_match = similarity_score * 100
        prompt += f"Filename: {filename}, Chunk ID: {chunk_id}, Similarity: {percentage_match:.2f}%\n"
        prompt += f"Text: {text}\n\n"

    prompt += "Provide a comprehensive answer to the question based on the above excerpts. Cite the sources by filename and chunk ID where applicable. If the excerpts do not provide enough information to answer the question, state that there is insufficient relevant data."

    return prompt

def generate_answer(prompt):
    """
    Generates an answer to the question using OpenAI's language model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return None

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
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

    logger.info("Displaying query results with similarity scores:")
    display_results(query_response)

    ai_prompt = prepare_ai_prompt(query_response, user_query)
    if not ai_prompt:
        logger.error("No data available to prepare AI prompt.")
        exit(1)

    logger.info("Generating answer using AI...")
    ai_answer = generate_answer(ai_prompt)
    if ai_answer:
        print("\n=== AI Generated Answer ===\n")
        print(ai_answer)
    else:
        print("Failed to generate an answer using the AI.")