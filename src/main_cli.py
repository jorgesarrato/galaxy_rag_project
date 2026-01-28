import os
from vectorize_pdf import IngestionPipeline
from retriever import Retriever
from generator import RAGGenerator

def main():
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
        
        gen = generator.generate_answer(query, context_chunks)
        tokens = []

        
        for token in gen:
            print(token, end="", flush=True)
            tokens.append(token)

        print("\n\n")

if __name__ == "__main__":
    main()
