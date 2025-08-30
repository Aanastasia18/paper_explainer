from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
import PyPDF2
import config
import gradio as gr

# ----- Define State -----
class PaperState(TypedDict):
    text: str
    summary: str
    explanation: str
    math_summary: str
    math_explanation: str

# ----- Initialize LLM -----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=config.API_KEY)

# ----- Define Node Functions -----
def summarize_node(state: PaperState) -> PaperState: 
    summary = llm.invoke(f"Summarize the following paper:\n\n{state['text']}")
    state["summary"] = summary.content
    return state

def explain_node(state: PaperState) -> PaperState:
    explanation = llm.invoke(f"Explain the main ideas in simple terms:\n\n{state['text']}")
    state["explanation"] = explanation.content
    return state

def math_summarize_node(state: PaperState) -> PaperState:
    math_summary = llm.invoke(f"Extract and summarize only the mathematical content and explain the math step by step in beginner-friendly language:\n\n{state['text']}")
    state["math_summary"] = math_summary.content
    return state

# def math_explain_node(state: PaperState) -> PaperState:
#     math_explanation = llm.invoke(
#         f"Explain the math step by step in beginner-friendly language:\n\n{state['math_summary']}"
#     )
#     state["math_explanation"] = math_explanation.content
#     return state

# ----- PDF Reader -----
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# ----- Build Graph -----
workflow = StateGraph(PaperState)

workflow.add_node("summarizer", summarize_node)
workflow.add_node("explainer", explain_node)
workflow.add_node("math_summarizer", math_summarize_node)
# workflow.add_node("math_explainer", math_explain_node)

workflow.set_entry_point("summarizer")

workflow.add_edge("summarizer", "explainer")
workflow.add_edge("explainer", "math_summarizer")
workflow.add_edge("math_summarizer", END)
# workflow.add_edge("math_explainer", END)

graph = workflow.compile()

# ----- Gradio App -----
def process_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file.name)
    result = graph.invoke({"text": text})

    # Wrap math formulas in $$ ... $$ for display
    math_summary_latex = "📊 " + result["math_summary"].replace(r"\[", "$$").replace(r"\]", "$$")
    # math_explanation_latex = "🔢 " + result["math_explanation"].replace(r"\[", "$$").replace(r"\]", "$$")

    # Wrap plain text outputs in Markdown for readability
    summary_md = "### 📝 Summary\n\n" + result["summary"]
    explanation_md = "### 💡 Explanation\n\n" + result["explanation"]

    return summary_md, explanation_md, math_summary_latex #, math_explanation_latex


with gr.Blocks() as demo:
    gr.Markdown("## 📄 Research Paper Analyzer with LangGraph Agents")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="filepath")
        
    with gr.Row():
        summary_output = gr.Markdown(label="📝 Summary") # , lines=8, interactive=False
        
    with gr.Row():
        explanation_output = gr.Markdown(label="💡 Explanation") # , lines=8, interactive=False
        
    with gr.Row():
        math_summary_output = gr.Markdown(label="📊 Math Summary")
        
    # with gr.Row():
    #     math_explanation_output = gr.Markdown(label="🔢 Math Explanation")
    
    pdf_input.change(
        process_pdf,
        inputs=[pdf_input],
        outputs=[summary_output, explanation_output, math_summary_output], # , math_explanation_output
    )

if __name__ == "__main__":
    demo.launch()
