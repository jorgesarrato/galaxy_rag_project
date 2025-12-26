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
