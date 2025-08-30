from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import PyPDF2
import config
from state import PaperState


llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=config.API_KEY)

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

# ----- Build Graph -----
def create_graph():
    workflow = StateGraph(PaperState)

    workflow.add_node("summarizer", summarize_node)
    workflow.add_node("explainer", explain_node)
    workflow.add_node("math_summarizer", math_summarize_node)

    workflow.set_entry_point("summarizer")

    workflow.add_edge("summarizer", "explainer")
    workflow.add_edge("explainer", "math_summarizer")
    workflow.add_edge("math_summarizer", END)

    graph = workflow.compile()

    return graph