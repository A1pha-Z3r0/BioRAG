"""This script has modules for loading and cleaning data"""

import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDF:
    def __init__(self, path):
        """Load PDF and initialize empty lists for chunks and cleaned chunks."""
        try:
            self.loader = PyMuPDFLoader(path)
            self.document = self.loader.load()
        except Exception as e:
            raise RuntimeError(f"Error loading the PDF: {e}")

        self.chunks = None
        self.cleaned_chunks = []

    def split_chunk(self, chunk_size=350, chunk_overlap=50): # 384 max token length for 'all-mpnet-base-v2'
        """Splits the document into chunks using RecursiveCharacterTextSplitter."""
        if not self.document:
            raise ValueError("No document loaded. Check the file path.")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.chunks = text_splitter.split_documents(self.document)
        except Exception as e:
            raise RuntimeError(f"Error splitting the document: {e}")

        if not self.chunks:
            raise ValueError("Chunking failed. No chunks created.")

    def clean_chunks(self):
        """Cleans the chunks by removing newlines and extra spaces."""
        if not self.chunks:
            raise ValueError("Chunks have not been created. Call split_chunk() first.")

        self.cleaned_chunks = [
            re.sub(r"\s+", " ", re.sub(r"\n", " ", chunk.page_content)).strip()
            for chunk in self.chunks
        ]

        return self.cleaned_chunks
