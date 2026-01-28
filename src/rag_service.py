import time
from retriever import Retriever
from generator import RAGGenerator

class RAGService:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = RAGGenerator()

    def answer(self, query: str, selected_papers=None) -> dict:
        if selected_papers is None:
            selected_papers = []

        t0 = time.perf_counter()
        chunks = self.retriever.get_relevant_context(query, selected_papers)
        retrieval_time = time.perf_counter() - t0

        gen = self.generator.generate_answer(query, chunks) # Can I just get the return of generate_answer?
        full_text = ""
        start_time = time.perf_counter()
        for token in gen:
            full_text += token
        generation_time = time.perf_counter() - start_time
    
        return {
            "answer": full_text,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
        }

