from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os

from config.config import EMBEDDING_MODEL, OUTPUT_FOLDER

class NLPProcessor:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.text_chunks = []

    def split_text(self, text, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.text_chunks = splitter.split_text(text)
        return self.text_chunks

    def build_faiss_index(self):
        embeddings = self.model.encode(self.text_chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self.index

    def query(self, query_text, top_k=3):
        query_emb = self.model.encode([query_text], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = [self.text_chunks[i] for i in I[0]]
        return results
