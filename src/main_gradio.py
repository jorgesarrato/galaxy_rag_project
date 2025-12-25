import gradio as gr
import os
from vectorize_pdf import IngestionPipeline
from retriever import Retriever
from generator import RAGGenerator
from utils.config import Config
import time

all_papers = [file_path.split("/")[-1] for file_path in os.listdir(Config.DATA_DIR)]

pipeline = IngestionPipeline()
pipeline.run()

retriever = Retriever()
generator = RAGGenerator()

def predict(message, history, selected_papers):
    start_retrieval = time.perf_counter()
    gr.Info("Searching papers...")
    chunks = retriever.get_relevant_context(message, selected_papers)
    retrieval_time = time.perf_counter() - start_retrieval
    
    final_text = ""
    gen_duration = 0
    
    gr.Info("Generating answer...")
    
    for partial_answer, duration in generator.generate_answer(message, chunks, app_mode = True):
        final_text = partial_answer
        if duration is not None:
            gen_duration = duration
        yield partial_answer

    stats_html = (
        f"\n\n---"
        f"\n⏱️ **Retrieval:** {retrieval_time:.2f}s | "
        f"**Generation:** {gen_duration:.2f}s | "
        f"total: {retrieval_time + gen_duration:.2f}s"
    )
    yield final_text + stats_html

examples = [
    ["What is the MAGICC project?", all_papers],
    ["Explain dark matter cored profiles.", all_papers]
]

demo = gr.ChatInterface(
    fn=predict,
    title="Astrophysics Research Assistant",
    description="Ask questions about galactic evolution and dark matter based on your local papers.",
    examples=examples,
    additional_inputs=[
        gr.CheckboxGroup(
            choices=all_papers, 
            value=all_papers,
            label="Source Papers",
            info="Select which papers to include in the search context."
        )
    ],
    additional_inputs_accordion="Filter Research Papers"
)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
