import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from utils.config import Config

class RAGGenerator:
    _instance = None 

    MODEL_MAP = {
        "Qwen2.5-7B-Instruct-Q8_0.gguf": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Qwen2.5-3B-Instruct-Q4_K_M.gguf": "bartowski/Qwen2.5-3B-Instruct-GGUF"
    }

    def __init__(self):
        self._ensure_model_exists()
        return

    def _ensure_model_exists(self):
        model_filename = Config.GENERATION_MODEL
        model_path = os.path.join(Config.MODEL_DIR, model_filename)

        if not os.path.exists(model_path):
            print(f"Model {model_filename} not found.")
            
            repo_id = self.MODEL_MAP.get(model_filename)
            if not repo_id:
                raise ValueError(f"No Repo ID mapped for {model_filename}. Update MODEL_MAP.")

            print(f"Downloading from Hugging Face: {repo_id}")
            
            _ = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                local_dir=Config.MODEL_DIR
                )
            print(f"Download complete.")
        
        return model_path

    def _load_model(self):
        if RAGGenerator._instance is None:
            model_path = self._ensure_model_exists()
            
            print(f"Loading {Config.GENERATION_MODEL}")
            RAGGenerator._instance = Llama(
                model_path=model_path,
                n_ctx=Config.LLM_CONTEXT,
                n_threads=Config.N_THREADS,
                n_batch=Config.N_BATCH,
                n_gpu_layers=Config.N_GPU_LAYERS,
                verbose=False,
                logits_all=False,
                use_mlock=True,
                use_mmap=True,

            )
        return RAGGenerator._instance


    def generate_answer(self, query, context_chunks):
        llm = self._load_model()

        context_text = ""
        for chunk in context_chunks:
            context_text += (
                f"\n---\n[Source: {chunk['source']}, Page: {chunk['page']}]\n"
                f"{chunk['text']}\n"
            )

        messages = [
            {"role": "system", "content": Config.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]

        import time
        start_time = time.time()

        stream = llm.create_chat_completion(
            messages=messages,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            max_tokens=Config.MAX_TOKENS,
            stream=True
        )

        full_text = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                full_text += token
                yield token

        elapsed = time.time() - start_time
        return full_text, elapsed
