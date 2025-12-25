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


        existing_files = set()
        metadata_path = os.path.join(Config.DB_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            self.index = faiss.read_index(f"{Config.DB_DIR}/docs.index")
            with open(metadata_path, "r", encoding="utf-8") as f:
                full_metadata = json.load(f)
                for item in full_metadata:
                    existing_files.add(item['metadata']['source'])
        
        chunks = []
        new_files = []

        all_source_files = os.listdir(Config.DATA_DIR)
        print(f"Found {len(all_source_files)} files in {Config.DATA_DIR}")

        for filename in all_source_files:
            if filename.endswith(".pdf"):
                if filename in existing_files:
                    print(f"Skipping {filename} (already indexed)")
                    continue
                
                new_files.append(filename)
                path = os.path.join(Config.DATA_DIR, filename)
                file_chunks = process_pdf(path)
                chunks.extend(file_chunks)

        if len(new_files) > 0:
            print(f"Found new files, adding to database: {new_files}")

            raw_texts = [item['text'] for item in chunks]
        
            print(f"Creating embeddings for {len(raw_texts)} chunks")
            embeddings = self.model.encode(raw_texts, show_progress_bar=True)
            
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings).astype('float32'))

            faiss.write_index(self.index, os.path.join(Config.DB_DIR, "docs.index"))

            if os.path.exists(metadata_path):
                full_metadata.extend(chunks)
            else:
                full_metadata = chunks

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, indent=4)
                    
            print(f"Vectors saved to: {os.path.join(Config.DB_DIR, 'docs.index')}")
            print(f"Metadata saved to: {metadata_path}")

        else:
            if not os.path.exists(metadata_path):
                print("No files found to create index")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()