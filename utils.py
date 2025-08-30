import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def format_math_steps(math_text: str) -> str:
    """
    Split the math summary into steps and wrap formulas in $$ for Markdown.
    """
    steps = math_text.split("\n\n")  # split by double line breaks
    formatted = ""
    for i, step in enumerate(steps, 1):
        step = step.replace(r"\[", "$$").replace(r"\]", "$$")
        formatted += f"**Step {i}:**\n{step}\n\n"
    return formatted