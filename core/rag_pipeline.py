from langchain.llms import Ollama
from core.nlp_processor import NLPProcessor

class RAGPipeline:
    def __init__(self, llm_model="llama2:7b"):
        self.llm = Ollama(model=llm_model)
        self.nlp = NLPProcessor()

    def query(self, text, question, top_k=3):
        self.nlp.split_text(text)
        self.nlp.build_faiss_index()
        retrieved_chunks = self.nlp.query(question, top_k=top_k)
        context = "\n".join(retrieved_chunks)
        prompt = f"Answer the question based on the context below:\nContext: {context}\nQuestion: {question}\nAnswer:"
        answer = self.llm(prompt)
        return answer
