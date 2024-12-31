from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import os
import logging
from werkzeug.utils import secure_filename
from pinecone_query_last_working import embed_query, query_pinecone, display_results
from pineconeupsert import embed_and_upsert_pdfs

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management and flash messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call the function to process and upsert the PDF
    try:
        embed_and_upsert_pdfs([file_path])
        return jsonify({'success': f'{filename} uploaded successfully'}), 200
    except Exception as e:
        logger.error(f'Error processing {filename}: {e}')
        return jsonify({'error': str(e)}), 500


# Route for the home page
@app.route('/home')
def home():
    return render_template('index.html')

# Route to handle PDF uploads
@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    files = request.files.getlist('file')
    if not files:
        flash('No selected file')
        return redirect(url_for('home'))

    file_paths = []
    for file in files:
        if file.filename == '':
            continue
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
            logger.info(f'File {filename} uploaded successfully.')

    if file_paths:
        # Call the function to embed and upsert PDFs
        embed_and_upsert_pdfs(file_paths)
        flash('Files uploaded and processed successfully.')
    else:
        flash('No valid PDF files uploaded.')
    return redirect(url_for('home'))

# Route to handle query submission
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query')
    if not user_query:
        flash('Please enter a query.')
        return redirect(url_for('home'))

    logger.info(f"Embedding the query: '{user_query}'")
    query_embedding = embed_query(user_query)
    if not query_embedding:
        flash('Failed to generate embedding for the query.')
        return redirect(url_for('home'))

    logger.info("Querying Pinecone for similar vectors...")
    query_response = query_pinecone(query_embedding, top_k=5)
    if not query_response:
        flash('Failed to retrieve query results from Pinecone.')
        return redirect(url_for('home'))

    results = []
    for match in query_response.matches:
        metadata = match.metadata
        filename = metadata.get("filename", "Unknown File")
        chunk_id = metadata.get("chunk_id", "Unknown Chunk")
        text = metadata.get("text", "No text available.")
        results.append({"filename": filename, "chunk_id": chunk_id, "text": text})

    return render_template('index.html', query_results=results)

# Route to list distinct filenames in the Pinecone index
@app.route('/list_files', methods=['GET'])
def list_files():
    try:
        logger.info("Querying Pinecone for distinct filenames...")
        # Using a filter to retrieve all vectors with a filename metadata
        response = pinecone_index.query(
            vector=[],  # Empty vector to match all
            top_k=1000,  # Adjust as needed
            include_values=False,
            include_metadata=True,
            namespace="ns1",
            filter={"filename": {"$exists": True}}
        )

        logger.info(f"Pinecone response: {response}")

        # Extract distinct filenames from metadata
        filenames = set()
        for match in response.matches:
            metadata = match.metadata
            filename = metadata.get("filename")
            if filename:
                filenames.add(filename)

        logger.info(f"Retrieved filenames: {filenames}")

        return render_template('index.html', filenames=filenames)
    except Exception as e:
        logger.error(f"Failed to list files in Pinecone: {e}")
        flash('Failed to retrieve file list from Pinecone.')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
