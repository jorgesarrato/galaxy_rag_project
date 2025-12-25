import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    DEBUG = os.getenv("ENV") == "development"
    
    MODEL_DIR = os.getenv("MODEL_DIR")
    
    GENERATION_MODEL = "Qwen2.5-3B-Instruct-Q4_K_M.gguf" 
    
    LLM_CONTEXT = 2048
    N_THREADS = 5
    N_BATCH = 1024
    N_GPU_LAYERS = 0
    
    TOP_P = 0
    TEMPERATURE = 0.0
    MAX_TOKENS = 512
    
    DATA_DIR = os.getenv("DATA_DIR")
    
    DB_DIR = os.getenv("DB_DIR")
    
    CHUNK_SIZE = 1000
    
    CHUNK_OVERLAP = 100
    
    N_CHUNKS_RETRIEVAL_INITIAL = 20
    N_CHUNKS_RETRIEVAL_FINAL = 3
    
    PARSER_TYPE = "pymupdf4llm"
    
    SYSTEM_PROMPT = """You are an expert Scientific Research Assistant. 

### CORE DIRECTIVES:
1. USE ONLY THE PROVIDED DATA: Your answer must be strictly based on the text provided between the <context> tags. 
2. CITATION FORMAT: ALWAYS, every factual claim must be followed by a citation in the format [Source, Page X], found in the context provided. 
3. UNCERTAINTY: If the context does not contain the specific information required to answer, state clearly: "I'm sorry, the provided research papers do not contain enough information to answer this question."

###EXAMPLE OF A GOOD ANSWER:
Based on the data, the formation of galactic discs is driven by the cooling of gas within dark matter haloes [2012_Magicc.pdf, Page 4]. However, some models suggest that stellar feedback can eject this gas before stars form [1986_Stellar_Feedback.pdf, Page 12].

FINAL CHECK: Before you output the answer, ensure every statement has a [Source, Page] citation. If no citations are present, the answer is invalid.

/no_think"""
