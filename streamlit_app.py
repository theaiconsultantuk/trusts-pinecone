import streamlit as st
from pineconeupsert import embed_and_upsert_pdfs
from pinecone_query_last_working import embed_query, query_pinecone
import openai
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the PDF directory to 'uploads'
pdf_directory = "uploads"

# Ensure the directory exists
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)

# OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set up the Streamlit app
def main():
    st.title("Pinecone PDF Processor")

    # Sidebar menu
    menu = ["Upload/Search", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload/Search":
        st.write("Upload your PDF files to process and index them in Pinecone.")

        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if st.button("Process Files"):
            if uploaded_files:
                file_paths = []
                for uploaded_file in uploaded_files:
                    # Save the uploaded file to the 'uploads' directory
                    file_path = os.path.join(pdf_directory, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                    st.write(f"Uploaded {uploaded_file.name} to {pdf_directory}")

                # Call the function to embed and upsert PDFs
                embed_and_upsert_pdfs(file_paths)
                st.success("Files processed and indexed successfully.")
            else:
                st.warning("Please upload at least one PDF file.")

        st.write("---")
        st.write("Search the indexed content.")
        search_query = st.text_input("Enter search query:")

        if st.button("Search"):
            if search_query:
                st.info(f"Embedding the query: '{search_query}'")
                query_embedding = embed_query(search_query)
                if not query_embedding:
                    st.error('Failed to generate embedding for the query.')
                else:
                    st.info("Querying Pinecone for similar vectors...")
                    query_response = query_pinecone(query_embedding, top_k=5)
                    if not query_response:
                        st.error('Failed to retrieve query results from Pinecone.')
                    else:
                        results = []
                        for match in query_response.matches:
                            metadata = match.metadata
                            filename = metadata.get("filename", "Unknown File")
                            chunk_id = metadata.get("chunk_id", "Unknown Chunk")
                            text = metadata.get("text", "No text available.")
                            score = match.score  # Similarity score
                            similarity_percentage = round(score * 100, 2)  # Convert to percentage
                            results.append({"filename": filename, "chunk_id": chunk_id, "text": text, "similarity": similarity_percentage})

                        if results:
                            st.write("Search Results:")
                            for result in results:
                                st.write(f"Filename: {result['filename']}, Chunk ID: {result['chunk_id']}")
                                st.write(f"Similarity: {result['similarity']}%")
                                st.write(f"Text: {result['text']}")
                                st.write("---")
                        else:
                            st.write("No results found.")
            else:
                st.warning("Please enter a search query.")

    elif choice == "Chatbot":
        st.write("Chat with the AI to explore the data.")
        user_input = st.text_input("You:", "Type your question here...")

        if st.button("Send"):
            if user_input:
                # Embed the user's query
                embedding = embed_query(user_input)

                # Query Pinecone for the top 3 similar results with error handling
                try:
                    search_results = query_pinecone(embedding, top_k=3)
                    if not search_results or not search_results.matches:
                        st.error("No results found in the index.")
                        logger.error("No results found in the index.")
                        return

                    # Log the top 3 matches with filename and text
                    logger.info("Top 3 matches:")
                    for match in search_results.matches:
                        filename = match['metadata'].get('filename', 'Unknown Filename')
                        text = match['metadata'].get('text', 'No text available')
                        logger.info(f"Score: {match['score']}, Filename: {filename}, Text: {text}")

                    # Evaluate the relevance of each result
                    relevant_insights = []
                    relevant_references = []
                    for i, result in enumerate(search_results.matches):
                        # Lower threshold for relevance
                        if result['score'] > 0.4:
                            text = result['metadata']['text']
                            filename = result['metadata'].get('filename', 'Unknown Filename')

                            # Log the evaluation prompt for debugging
                            logger.info(f"Evaluation Prompt: Query: {user_input}\nText: {text}")

                            # Use AI to evaluate if the text is useful as a boolean
                            evaluation_prompt = (
                                "You are an expert in evaluating the relevance of documents in an index against a user's query. "
                                "Please respond with 'true' if the text is useful for answering the user's query, otherwise respond with 'false'. "
                                f"Query: {user_input}\nText: {text}"
                            )

                            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                            evaluation_response = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": "You are an expert in evaluating text relevance."},
                                    {"role": "user", "content": evaluation_prompt}
                                ],
                                model="gpt-4o",
                                max_tokens=10  # Small token count for boolean response
                            )
                            evaluation_result = evaluation_response.choices[0].message.content.strip().lower()

                            if evaluation_result == "true":
                                # Extract relevant parts using AI
                                extraction_prompt = (
                                    "You are an expert in extracting relevant information from documents. "
                                    "Please extract the parts of the text that are most relevant to the query. "
                                    f"Query: {user_input}\nText: {text}"
                                )

                                extraction_response = client.chat.completions.create(
                                    messages=[
                                        {"role": "system", "content": "You are an expert in extracting information."},
                                        {"role": "user", "content": extraction_prompt}
                                    ],
                                    model="gpt-4o",
                                    max_tokens=150  # Increase tokens for extraction
                                )
                                extracted_text = extraction_response.choices[0].message.content.strip()
                                relevant_insights.append(extracted_text)
                                relevant_references.append(filename)

                    if not relevant_insights:
                        st.info("No suitable data was found in the index. Providing a general answer.")
                        # Log the text from sources
                        for match in search_results.matches:
                            filename = match['metadata'].get('filename', 'Unknown Filename')
                            text = match['metadata'].get('text', 'No text available')
                            logger.info(f"Filename: {filename}, Text: {text}")

                        # Provide a general answer using AI
                        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                        response = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are an expert in making complex concepts easy to understand. You are also a top teacher with a great command of the English language."},
                                {"role": "user", "content": user_input}
                            ],
                            model="gpt-4o",
                            max_tokens=300
                        )
                        answer = response.choices[0].message.content.strip()
                        st.write("AI:", answer)
                    else:
                        # Enhance the prompt with persona and relevant search results
                        enhanced_prompt = (
                            "You are an expert in making complex concepts easy to understand. "
                            "You are also a top teacher with a great command of the English language. "
                            "Your task is to explain the following based on the user's query and relevant search results: "
                            f"{user_input}. Here are some relevant insights: "
                        )
                        for i, insight in enumerate(relevant_insights):
                            enhanced_prompt += f"{i+1}. {insight} (Filename: {relevant_references[i]}) "

                        # Call OpenAI GPT-4 to get a response using the new SDK interface
                        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                        response = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are an expert in making complex concepts easy to understand. You are also a top teacher with a great command of the English language."},
                                {"role": "user", "content": enhanced_prompt}
                            ],
                            model="gpt-4o",
                            max_tokens=300
                        )
                        answer = response.choices[0].message.content.strip()
                        st.write("AI:", answer)

                        # Add references at the bottom of the answer
                        st.write("# References")
                        for ref in relevant_references:
                            st.write(f"- {ref}")

                        # Provide a way to view full context using a modal
                        if st.button("View Full Context"):
                            st.write("## Full Context")
                            for i, ref in enumerate(relevant_references):
                                st.write(f"**Source {i+1}: {ref}**")
                                st.write(relevant_insights[i])
                except Exception as e:
                    st.error("An error occurred while querying the index.")
                    logger.error(f"Error querying Pinecone: {e}")
                    return

            else:
                st.warning("Please enter a message.")

if __name__ == "__main__":
    main()
