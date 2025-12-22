import re
from PyPDF2 import PdfReader
from utils.config import Config

def clean_scientific_text(text):
    # Fix artificially broken words
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
    
    text = re.sub(r'\s+', ' ', text) # Normalize spaces
    
    return text.strip()
    
def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + Config.CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += (Config.CHUNK_SIZE - Config.CHUNK_OVERLAP)
    return chunks

def process_pdf(file_path):
    reader = PdfReader(file_path)
    filename = file_path.split("/")[-1]
    chunks = []
    
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text: continue
        
        clean_text = clean_scientific_text(raw_text)
        text_chunks = chunk_text(clean_text)
        
        for chunk in text_chunks:
            chunks.append({"text": chunk,"metadata": {"source": filename, "page": i + 1}})
            
    return chunks
