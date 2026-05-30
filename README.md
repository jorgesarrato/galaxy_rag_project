# Galaxy RAG Project

A CPU-optimized Retrieval-Augmented Generation (RAG) system for analyzing scientific papers.

[Example_Recording.webm](https://github.com/user-attachments/assets/64053713-26b3-4ba8-bf47-c9ce81cc929b)

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

## Data Placing

Store your PDF files in data/ or the folder you defined in your .env as DATA_DIR

## Usage

Execute in terminal mode:

```bash
python src/main.py
```

Or in app mode, and open the provided local link to chat:

```bash
python src/main_gradio.py
```

## Running with Docker

The project can also be run as a **containerized service**, which avoids local dependency issues and ensures reproducibility across systems.

### Prerequisites

1. Copy the environment template and add your Hugging Face token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN to your Hugging Face token
```

2. Download the LLM model:

```bash
mkdir -p llm_models
huggingface-cli download bartowski/Qwen2.5-3B-Instruct-GGUF \
  --include "Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
  --local-dir llm_models
```

3. Place your PDF files in the `data/` directory.

### Run with Docker Compose (recommended)

```bash
docker compose up --build
```

This will:
- Build the image with your PDFs baked in
- Mount `llm_models/` and `vectors/` from your host
- Run ingestion automatically if no vector index exists
- Start the API on port 8000

### Run with Docker manually

Build the image:

```bash
docker build -t rag-api .
```

Run the container:

```bash
docker run -d -p 8000:8000 \
  --env-file .env \
  -e DATA_DIR=data \
  -e DB_DIR=vectors \
  -e MODEL_DIR=llm_models \
  -v ./llm_models:/app/llm_models \
  -v ./vectors:/app/vectors \
  --name rag-api \
  rag-api
```

### Health Check

Verify the service is running:

```bash
curl http://localhost:8000/health
```

### Query the API

Send a query with specific papers:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the physical motivation for cored dark matter profiles",
    "selected_papers": [
      "Rocha_2013_SIDM.pdf",
      "Kaplinghat_2016_DMcores.pdf",
      "Bullock_2017_CDMreview.pdf"
    ]
  }'
```

Or query across all papers:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain abundance matching"}'
```



