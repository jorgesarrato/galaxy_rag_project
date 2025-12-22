import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from utils.config import Config

class IngestionPipeline:
    def __init__(self):
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.index = None

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

    def chunk_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + Config.CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            start += (Config.CHUNK_SIZE - Config.CHUNK_OVERLAP)
        return chunks

    def run(self):
        chunks = []
        
        for filename in os.listdir(Config.DATA_DIR):
            if filename.endswith(".pdf"):
                print(f"Processing {filename}")
                path = os.path.join(Config.DATA_DIR, filename)
                raw_text = self.extract_text_from_pdf(path)
                chunks_temp = self.chunk_text(raw_text)
                chunks.extend(chunks_temp)

        print(f"Creating embeddings for {len(chunks)} chunks.")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        faiss.write_index(self.index, os.path.join(Config.DB_DIR, "docs.index"))
        
        with open(os.path.join(Config.DB_DIR, "chunks.txt"), "w", encoding="utf-8") as f:
            for chunk in chunks:
                # Remove newlines for storage consistency
                clean_chunk = chunk.replace("\n", " ")
                f.write(clean_chunk + "\n")
                
        print(f"Vectorization complete. Database saved to {Config.DB_DIR}")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()