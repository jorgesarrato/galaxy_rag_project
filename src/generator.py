import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from utils.config import Config

class RAGGenerator:
    _instance = None 

    MODEL_MAP = {
        "Qwen2.5-7B-Instruct-Q8_0.gguf": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    }

    def __init__(self):
        pass

    def _ensure_model_exists(self):
        model_filename = Config.GENERATION_MODEL
        model_path = os.path.join(Config.MODEL_DIR, model_filename)

        if not os.path.exists(model_path):
            print(f"Model {model_filename} not found.")
            
            repo_id = self.MODEL_MAP.get(model_filename)
            if not repo_id:
                raise ValueError(f"No Repo ID mapped for {model_filename}. Update MODEL_MAP.")

            print(f"Downloading from Hugging Face: {repo_id}")
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                local_dir=Config.MODEL_DIR,
                local_dir_use_symlinks=False 
            ) # For some reason download is too slow, manual download is faster. Need to fix.
            print(f"Download complete.")
        
        return model_path

    def _load_model(self):
        if RAGGenerator._instance is None:
            model_path = self._ensure_model_exists()
            
            print(f"Loading {Config.GENERATION_MODEL}")
            RAGGenerator._instance = Llama(
                model_path=model_path,
                n_ctx=Config.LLM_CONTEXT,
                n_threads=12,
                verbose=False
            )
        return RAGGenerator._instance

    def generate_answer(self, query, context_chunks):
        llm = self._load_model()
        
        context_text = ""
        for chunk in context_chunks:
            context_text += f"\n---\n[Source: {chunk['source']}, Page: {chunk['page']}]\n"
            context_text += f"Content: {chunk['text']}\n"

        messages = [
            {"role": "system", "content": """You are a scientific assistant. Use the context to answer precisely. Cite paper and page.
             If context doesn't contain enoug information to answer the question, say you don't know."""},
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
        ]

        response = llm.create_chat_completion(messages=messages, temperature=0.1)
        return response["choices"][0]["message"]["content"]