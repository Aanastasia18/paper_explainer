import gradio as gr
from utils import extract_text_from_pdf, format_math_steps
from graph import create_graph

graph = create_graph()

def process_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file.name)
    result = graph.invoke({"text": text})

    math_summary_latex = "### 📊 Math Summary\n\n" + format_math_steps(result["math_summary"])
    summary_md = "### 📝 Summary\n\n" + result["summary"]
    explanation_md = "### 💡 Explanation\n\n" + result["explanation"]

    return summary_md, explanation_md, math_summary_latex

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Research Paper Analyzer with LangGraph Agents")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="filepath")
        
    with gr.Row():
        summary_output = gr.Markdown(label="📝 Summary")
        
    with gr.Row():
        explanation_output = gr.Markdown(label="💡 Explanation")
        
    with gr.Row():
        math_summary_output = gr.Markdown(label="📊 Math Summary")
    
    pdf_input.change(
        process_pdf,
        inputs=[pdf_input],
        outputs=[summary_output, explanation_output, math_summary_output],
    )

if __name__ == "__main__":
    demo.launch()
