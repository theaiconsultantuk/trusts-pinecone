import streamlit as st
from pineconeupsert import embed_and_upsert_pdfs
from pinecone_query_last_working import embed_query, query_pinecone
import openai
import os

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
                # Call OpenAI GPT-4 to get a response
                response = openai.Completion.create(
                    engine="gpt-4",
                    prompt=user_input,
                    max_tokens=150
                )
                answer = response.choices[0].text.strip()
                st.write("AI:", answer)
            else:
                st.warning("Please enter a message.")

if __name__ == "__main__":
    main()
