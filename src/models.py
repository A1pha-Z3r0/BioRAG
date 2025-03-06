import torch

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

class MODELS:
    def __init__(self,):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embeddings = []

    def generate_embeddings(self, cleaned_chunks):

        for chunk in cleaned_chunks:
            chunk_embedding = self.model.encode(chunk)
            self.embeddings.append(chunk_embedding)

        print("Finished generating embeddings")


    def save_faiss(self, index_file):

        if not self.embeddings:
            raise ValueError("No embeddings found")

        embeddings_np = np.vstack(self.embeddings)

        # Use a FAISS index for L2 distance (cosine similarity)
        index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Flat index for L2 distance
        index.add(embeddings_np)  # Add embeddings to the index

        # Save the index to a file
        faiss.write_index(index, index_file)
        print(f"FAISS index saved to {index_file}")

