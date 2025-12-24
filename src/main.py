import os
from vectorize_pdf import IngestionPipeline
from retriever import Retriever
from generator import RAGGenerator
from utils.config import Config

def main():
    index_path = os.path.join(Config.DB_DIR, 'docs.index')
    
    if not os.path.exists(index_path):
        print(" Index not found. Starting Vectorizing papers.")
        pipeline = IngestionPipeline()
        pipeline.run()
    else:
        print("Existing Index found. Loading.")

    retriever = Retriever()
    generator = RAGGenerator()

    print("\n" + "="*50)
    print("ASTROPHYSICS RAG ASSISTANT READY")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        query = input("Question: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Closing session. Bye!")
            break
            
        if not query.strip():
            continue
        
        print("Searching papers...")
        context_chunks = retriever.get_relevant_context(query)
        
        print("Generating answer...")
        answer = generator.generate_with_stats(query, context_chunks)
        
        print("\n" + "-"*30)
        print(f"ANSWER:\n{answer}")
        print("-"*30 + "\n")

if __name__ == "__main__":
    main()
