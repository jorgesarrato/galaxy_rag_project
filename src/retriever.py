import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import Config

class Retriever:
    def __init__(self):
        print("Loading chunk database")
        self.model = SentenceTransformer(Config.MODEL_NAME)
        
        self.index = faiss.read_index(f"{Config.DB_DIR}/docs.index")
        
        with open(f"{Config.DB_DIR}/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def get_relevant_context(self, query, top_n_chunks=Config.N_CHUNKS_RETRIEVAL):
        query_vec = self.model.encode([query])
        
        # Euclidean distance
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), top_n_chunks)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if it doesn't find matches
                item = self.metadata[idx]
                results.append({
                    "text": item['text'],
                    "source": item['metadata']['source'],
                    "page": item['metadata']['page'],
                    "score": float(distances[0][i])
                })
        
        return results

    def format_context(self, retrieved_results):
        context_blocks = []
        for res in retrieved_results:
            block = f"[Source: {res['source']}, Page: {res['page']}]\n{res['text']}"
            context_blocks.append(block)
        
        return "\n\n---\n\n".join(context_blocks)

if __name__ == "__main__":
    r = Retriever()
    query = "What galaxies are more likely to present cored dark matter profiles?"
    results = r.get_relevant_context(query)
    
    print(f"\nQUERY: {query}")
    print("\nRETRIEVED CONTEXT:")
    print(r.format_context(results))