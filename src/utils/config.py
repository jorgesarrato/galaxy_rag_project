import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    DEBUG = os.getenv("ENV") == "development"
    
    MODEL_DIR = os.getenv("MODEL_DIR")
    
    GENERATION_MODEL = "Qwen2.5-3B-Instruct-Q4_K_M.gguf" 
    
    LLM_CONTEXT = 2048
    
    DATA_DIR = os.getenv("DATA_DIR")
    
    DB_DIR = os.getenv("DB_DIR")
    
    CHUNK_SIZE = 1000
    
    CHUNK_OVERLAP = 100
    
    N_CHUNKS_RETRIEVAL = 2
    
    PARSER_TYPE = "pymupdf4llm"
