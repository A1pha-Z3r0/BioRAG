from pdf_processing import PDF
from models import MODELS
from query_script import QueryScript
from transformers import pipeline



def main():
    """This has the main business logic."""
    # Initialize PDF class with the path to your PDF file
    pdf = PDF("../data/biochem.pdf")

    # Split PDF into chunks
    pdf.split_chunk()

    # list of all cleaned texts
    cleaned_text = pdf.clean_chunks()

    # Initialize the MODELS class (no argument in the constructor)
    model = MODELS()

    # Generate embeddings using the cleaned text chunks
    model.generate_embeddings(cleaned_text)

    # Save the FAISS index

    index_file_path = "faiss_index.index"
    model.save_faiss(index_file_path)

    # Define your query (you can take input from user as well)
    #query = input("Enter your question: ")

    query = ("where does transcription occurs in eukaryotes")

    queryScript = QueryScript(index_file_path)

    # Embed the query
    queryScript.embed_query([query])  # Passing the query as a list for consistency

    # Search for similar chunks in the FAISS index
    closest_paragraphs = queryScript.search_index(cleaned_text)

    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Step 7: Answer Extraction
    answers = []
    for chunk in closest_paragraphs:
        result = qa_pipeline(question=query, context=chunk)  # Extract answer from chunk
        answers.append(result['answer'])  # Append the answer

    # Step 8: Output the Answers
    print("\nQuestion:", query)
    print("\nTop Answers:")
    for i, (answer, chunk) in enumerate(zip(answers, closest_paragraphs)):
        print(f"{i + 1}. {answer}")
        print(f"   (from: {chunk[:150]}...)\n")  # Show a snippet of the chunk



if __name__ == "__main__":
    main()
# how many percentage human genome codes for protein
