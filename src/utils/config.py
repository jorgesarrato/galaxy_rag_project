import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    DEBUG = os.getenv("ENV") == "development"
    
    GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    
    DATA_DIR = os.getenv("DATA_DIR")
