import faiss
from models import MODELS


class QueryScript:
    def __init__(self,index_path):
        self.faiss_file = faiss.read_index(index_path)
        print(f"FAISS index loaded from {index_path}.")
        self.query_embedding = None

    def embed_query(self, query):
        model = MODELS()

        self.query_embedding = model.model.encode(query)
        return self.query_embedding

    def search_index(self, cleaned_chunks):
        faiss.normalize_L2(self.query_embedding)
        distances, indices = self.faiss_file.search(self.query_embedding, 5)
        top_chunks = [cleaned_chunks[idx] for idx in indices[0]]
        return top_chunks




