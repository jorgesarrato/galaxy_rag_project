import gradio as gr
import os
from vectorize_pdf import IngestionPipeline
from retriever import Retriever
from generator import RAGGenerator
from utils.config import Config
import time

all_papers = sorted([file_path.split("/")[-1] for file_path in os.listdir(Config.DATA_DIR)])

pipeline = IngestionPipeline()
pipeline.run()

retriever = Retriever()
generator = RAGGenerator()

custom_css = """
#paper-scroll-container {
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 8px;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("# Astrophysics Research Assistant")
    
    with gr.Row():
        with gr.Column(scale=3) as left_column:
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask questions about galactic evolution and dark matter based on your local papers.")
            
        with gr.Column(scale=1):
            gr.Markdown("### Source Selection")
            
            with gr.Row():
                btn_all = gr.Button("Select All", size="sm")
                btn_none = gr.Button("Unselect All", size="sm")
            
            with gr.Column(elem_id="paper-scroll-container"): 
                paper_selector = gr.CheckboxGroup(
                    choices=all_papers,
                    value=all_papers,
                    label="Selected Papers",
                    interactive=True
                )
                
        with left_column:
            gr.Examples(
                examples=[
                    ["What is the MAGICC project?", all_papers],
                    ["Explain cored dark matter profiles.", all_papers]
                ],
                inputs=[msg, paper_selector]
            )
            clear = gr.ClearButton([msg, chatbot])

    def select_all_papers():
        return gr.update(value=all_papers)

    def unselect_all_papers():
        return gr.update(value=[])

    btn_all.click(fn=select_all_papers, outputs=paper_selector)
    btn_none.click(fn=unselect_all_papers, outputs=paper_selector)

    def predict(message, history, selected_papers):
        if not selected_papers:
            history.append({"role": "assistant", "content": [{"type": "text", "text": "Please select at least one paper."}]})
            yield history
            return
    
        start_retrieval = time.perf_counter()
        gr.Info("Searching papers...")
        chunks = retriever.get_relevant_context(message, selected_papers)
        retrieval_time = time.perf_counter() - start_retrieval
            
        final_text = ""
        gen_duration = 0
        gr.Info("Generating answer...")
       
        history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
            
        for partial_answer, duration in generator.generate_answer(message, chunks, app_mode=True):
            history[-1]["content"][0]["text"] = partial_answer
            if duration is not None:
                gen_duration = duration
            yield history

        stats_html = (
            f"\n\n---"
            f"\n⏱️ **Retrieval:** {retrieval_time:.2f}s | "
            f"**Generation:** {gen_duration:.2f}s | "
            f"**Total:** {retrieval_time + gen_duration:.2f}s"
        )
        
        history[-1]["content"][0]["text"] += stats_html
        yield history
        
    msg.submit(
        fn=lambda message, history: history + [{"role": "user", "content": [{"type": "text", "text": message}]}],
        inputs=[msg, chatbot],
        outputs=[chatbot],
        queue=False
    ).then(
        fn=predict,
        inputs=[msg, chatbot, paper_selector],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, theme='glass', css=custom_css)
