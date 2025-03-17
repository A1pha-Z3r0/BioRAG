"""This script has the main logic for a QnA bot for a pdf"""

import argparse

from pdf_processing import PDF
from models import MODELS
from query_script import QueryScript

def main(pdf_path):
    """This has the main business logic."""
    # Initialize PDF class with the path to your PDF file
    pdf = PDF(pdf_path)

    # Split PDF into chunks
    pdf.split_chunk()

    # list of all cleaned texts
    cleaned_text = pdf.clean_chunks()

    # Initialize the MODELS class
    model = MODELS()

    # Generate embeddings using the cleaned text chunks
    model.generate_embeddings(cleaned_text)

    # Save the FAISS index
    index_file_path = "faiss_index.index"
    model.save_faiss(index_file_path)

    # Enter query
    query = input("Enter your question: ")

    # Load FAISS index for query
    query_script = QueryScript(index_file_path)

    # Embed the query
    query_script.embed_query([query])  # Passing the query as a list for consistency

    # Search for similar chunks in the FAISS index
    closest_paragraphs = query_script.search_index(cleaned_text)

    # Send relevant chunks as a paragraph to an LLM (here mistral)
    relevant_chunks = " ".join(closest_paragraphs)

    # Generate response
    response = model.response_generation(query,relevant_chunks)

    # Print response
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QnA Bot for PDF using RAG.")
    parser.add_argument("--pdf_path","-p", type=str, help="Path to the PDF file")

    args = parser.parse_args()

    main(args.pdf_path)
