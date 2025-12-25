import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from utils.config import Config

class Retriever:
    def __init__(self):
        print("Loading chunk database...")
        self.model = SentenceTransformer(Config.MODEL_NAME)

        self.reranker = CrossEncoder(Config.RERANKER_MODEL_NAME)
        
        self.index = faiss.read_index(f"{Config.DB_DIR}/docs.index")
        
        with open(f"{Config.DB_DIR}/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def get_initial_relevant_context(self, query, top_n_chunks=Config.N_CHUNKS_RETRIEVAL_INITIAL):
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

    def get_relevant_context(self, query):
            initial_results = self.get_initial_relevant_context(query, top_n_chunks=Config.N_CHUNKS_RETRIEVAL_INITIAL) 
            
            pairs = [[query, res['text']] for res in initial_results]
            
            scores = self.reranker.predict(pairs)
            
            for i, res in enumerate(initial_results):
                res['rerank_score'] = scores[i]
                
            initial_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return initial_results[:Config.N_CHUNKS_RETRIEVAL_FINAL]

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