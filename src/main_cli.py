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

        
        it = iter(gen)
        while True:
            try:
                token = next(it)
                print(token, end="", flush=True)
                tokens.append(token)
            except StopIteration as e:
                result = e.value 
                if result:
                    answer, elapsed = result
                    print(f"\nElapsed time: {round(elapsed)}s\n\n")
                break

if __name__ == "__main__":
    main()
