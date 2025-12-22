import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import Config
from utils.text_processing import process_pdf

class IngestionPipeline:
    def __init__(self):
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.index = None

    def run(self):
        chunks = []
        
        for filename in os.listdir(Config.DATA_DIR):
            if filename.endswith(".pdf"):
                print(f"Processing: {filename}")
                path = os.path.join(Config.DATA_DIR, filename)
                
                file_chunks = process_pdf(path)
                chunks.extend(file_chunks)

        if not chunks:
            print(f"No PDF data found! Check your data directory: {Config.DATA_DIR}")
            return

        raw_texts = [item['text'] for item in chunks]
        
        print(f"Creating embeddings for {len(raw_texts)} chunks")
        embeddings = self.model.encode(raw_texts, show_progress_bar=True)
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

        faiss.write_index(self.index, os.path.join(Config.DB_DIR, "docs.index"))
        
        metadata_path = os.path.join(Config.DB_DIR, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4)
                
        print(f"Vectors saved to: {os.path.join(Config.DB_DIR, 'docs.index')}")
        print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()