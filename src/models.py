"""This script has modules related to model loading and answer generation"""

import faiss
import numpy as np
import os

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

os.environ["OMP_NUM_THREADS"] = "1"

class MODELS:
    def __init__(self,):
        """Load model and initialize 'embeddings' variable as None"""
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embeddings = None

    def generate_embeddings(self, cleaned_chunks):
        """Generate embeddings for cleaned chunks"""
        self.embeddings = np.array(self.model.encode(cleaned_chunks))
        print("Finished generating embeddings")


    def save_faiss(self, index_file):
        """Save FAISS embeddings file """
        if self.embeddings is None: # or self.embeddings.size == 0:
            raise ValueError("No embeddings found")

        # Use a FAISS index for L2 distance (cosine similarity)
        index = faiss.IndexFlatL2(self.embeddings.shape[1])  # Flat index for L2 distance
        index.add(self.embeddings)  # Add embeddings to the index

        # Save the index to a file
        faiss.write_index(index, index_file)
        print(f"FAISS index saved to {index_file}")
        del index

    def llm_prompt(self):
        """Create prompt for the LLM"""
        template = """
        You are a Question and Answer bot that helps people understand complex pdf, given context is the top relevant 
        chunks for the question asked by the user. Understand these chunks and answer the question given below. 
        If you don't know please answer 3"I DON'T KNOW, sorry".

        context: {context}

        Question: {question}

        Answer:

        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt
    
    def response_generation(self,question,context):
        """Generate response through llm"""
        # Currently using mistral since it's a little uncensored.
        llm_model = OllamaLLM(model="mistral")
        prompt = self.llm_prompt()
        chain = prompt | llm_model
        
        response_generation = chain.invoke({"context": context,"question": question})
        return response_generation
