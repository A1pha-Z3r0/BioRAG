from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import re

# Step 1: Load Models
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # For embeddings
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")  # For Q&A

# Step 2: Define and Split Large Text
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    """
    Split text into chunks of a specified size with overlap.
    """
    sentences = re.split(r'\.|\n', text)  # Split by sentences or line breaks
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        current_length += len(sentence)
        current_chunk.append(sentence)
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap // len(sentence):]  # Add overlap
            current_length = len(" ".join(current_chunk))

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Replace this with your large block of text
large_text = """
Hello, one guy's name is jonathan, he's an asshole. He has a girlfriend named bondhala. He loves her so much but sometimes 
he is not considerate her. He doesn't make any sense while talking. He is 79 years old and loves his mom. He wears only white
t-shirts. His underwear size is medium. He has a very good sense of taste in girls but he made an exception once. His 
favourite girl in the world after his mom is alia bhatt. He hates onion and curd and tomatoes because he cant stand the smell
of them. Those things make him puke.
"""

# Split text into chunks
chunks = split_text_into_chunks(large_text)

# Step 3: Embed Chunks
chunk_embeddings = embedding_model.encode(chunks)

# Step 4: Create FAISS Index
dimension = chunk_embeddings.shape[1]  # Embedding dimension (typically 384 for MiniLM)
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
index.add(chunk_embeddings)  # Add embeddings to FAISS index

# Step 5: Question Input
question = input("Enter your question: ")  # User input for the question
question_embedding = embedding_model.encode([question])  # Embed the question

# Step 6: Retrieve Top Chunks
k = 3  # Number of top chunks to retrieve
distances, indices = index.search(question_embedding, k=k)  # FAISS search
top_chunks = [chunks[idx] for idx in indices[0]]  # Get the top chunks

# Step 7: Answer Extraction
answers = []
for chunk in top_chunks:
    result = qa_pipeline(question=question, context=chunk)  # Extract answer from chunk
    answers.append(result['answer'])  # Append the answer

# Step 8: Output the Answers
print("\nQuestion:", question)
print("\nTop Answers:")
for i, (answer, chunk) in enumerate(zip(answers, top_chunks)):
    print(f"{i+1}. {answer}")
    print(f"   (from: {chunk[:150]}...)\n")  # Show a snippet of the chunk