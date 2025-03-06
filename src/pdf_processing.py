"""
This script has modules for loading and cleaning data
"""

import re
import sys


from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDF:
    def __init__(self,path):
        """ This loads the pdf; and initializes an empty array for cleaned chunks."""
        #load the pdf
        try:
            self.loader = PyMuPDFLoader(path)
            self.document = self.loader.load()
        except Exception as e:
            print(f"Error with loading the pdf: {e}.")
            sys.exit(1)

        self.chunks = None
        self.cleaned_chunks = []


    def split_chunk(self,chunk_size = 700, chunk_overlap = 70):
        """This splits the chunks based on recursive character splitting."""

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.chunks = text_splitter.split_documents(self.document)

            self.chunk_to_paragraph_map = [
                chunk.page_content for chunk in self.chunks
            ]


        except Exception as e:
            print(f"Error with splitting the chunk: {e}.")
            sys.exit(1)

        if self.chunks is None:
            raise ValueError("Chunks has not been loaded/ created..")


    def clean_chunks(self,):
        if self.cleaned_chunks is None:
            raise ValueError("Chunks have not been created. Call split_chunk() first.")

        self.cleaned_chunks = []

        for chunk in self.chunks:
            text = re.sub(r"\n", " ", chunk.page_content)
            text = re.sub(r"\s+", " ", text).strip()

            self.cleaned_chunks.append(text)

        return self.cleaned_chunks