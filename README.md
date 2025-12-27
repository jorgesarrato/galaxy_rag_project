# Galaxy RAG Project

A CPU-optimized Retrieval-Augmented Generation (RAG) system for analyzing scientific papers.

## Features

- **CPU Optimized:** Runs efficiently on local hardware (tested on 6 physical cores) using `Qwen2.5-3B` (GGUF). Answers take 15-30 seconds to generate.
- **Intelligent Retrieval:** Vector search using FAISS and a reranker model.
- **Layout-Aware Parsing:** Handles multi-column scientific PDFs without header/footer noise. Text is recurively split into chunks.
- **Incremental Indexing:** Only processes new PDFs added to the data directory.
- **Verified Citations:** Instructed to include precise references to consulted documents.
- **Stream Generation:** Improved latency feeling by printing each token right after generation.
- **User Interface:** Gradio chatbot interface.
- **Paper Selection:** If desired, you can choose specific papers to allow for retrieval.

## Models

- **LLM:** Qwen2.5-3B-Instruct (Quantized Q4_K_M) via `llama-cpp-python`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker** `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Installation

This project is built using **Python 3.10.12**. Use a virtual environment to avoid conflicts.

```bash
git clone https://github.com/jorgesarrato/galaxy_rag_project.git
cd rag_project
```

If you want to use a virtual environment:

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Create a .env file with the following content:

```bash
HF_TOKEN=#Your huggingface token
DATA_DIR=#Path to PDFs
DB_DIR=#Path to store vector database
MODEL_DIR=#Path to llm models
```

Store llm models:

```bash
mkdir # your MODEL_DIR
hf download bartowski/Qwen2.5-3B-Instruct-GGUF --include "Qwen2.5-3B-Instruct-Q4_K_M.gguf" --local-dir # your MODEL_DIR
```

Theoretically the pipeline will download your model if you include it in the MODEL_MAP.
In practice I found it's faster to call hf download manually.

## Usage

Execute in terminal mode:

```bash
python src/main.py
```

Or in app mode, and open the provided local link to chat:

```bash
python src/main_gradio.py
```
