import os
import sys
import time
from reportlab.pdfgen import canvas
from agent.tools import extract_and_summarize_chunk, TOOL_REGISTRY
from main import run_agentic_loop

def generate_demo_pdf(filename="demo_input.pdf"):
    c = canvas.Canvas(filename)
    # Give it enough words to guarantee 3-4 chunks (assuming chunk size ~40)
    text = (
        "Agentic loops are a new paradigm in AI engineering. They allow an LLM to iteratively perceive its environment, "
        "reason about what to do next, take actions using tools, and reflect on the outcome. "
        "The perception stage structures raw data into an observation. The reasoning stage acts as the brain, formulating a plan. "
        "The action stage executes the plan by interacting with the world. "
        "Finally, the reflection stage evaluates if the goal was met. If not, it provides feedback to the reasoning stage to try again. "
        "This process continues until the task is complete or max iterations reached. "
        "Memory plays a critical role. A short-term session memory prevents infinite loops by remembering recent actions. "
        "Long-term semantic memory allows the agent to recall past summaries and synthesize information coherently. "
        "By combining these elements, we achieve a robust, autonomous AI system capable of complex document analysis."
    )
    words = text.split()
    
    y = 750
    line = []
    for word in words:
        line.append(word)
        if len(line) > 10:
            c.drawString(100, y, " ".join(line))
            y -= 20
            line = []
    if line:
        c.drawString(100, y, " ".join(line))
        
    c.save()
    print(f"Created {filename} with {len(words)} words.")

# Chaos Monkey for Tool Failure
original_tool = extract_and_summarize_chunk
call_count = 0

def chaotic_extract_and_summarize_chunk(filepath, start_word_idx, end_word_idx):
    global call_count
    call_count += 1
    # Fail intentionally on the second chunk to demonstrate graceful failure
    if call_count == 2:
        print("\n[CHAOS MONKEY] Simulating a tool failure during text extraction!\n")
        raise ValueError("Simulated PDF parsing error. Network timeout or corrupted data.")
    return original_tool(filepath, start_word_idx, end_word_idx)

if __name__ == "__main__":
    print("=== Agentic Loop Demo Script ===")
    pdf_file = "demo_input.pdf"
    
    # 1. Generate a PDF large enough to trigger multiple iterations
    generate_demo_pdf(pdf_file)
    
    # 2. Inject chaotic tool to demonstrate graceful failure handling
    print("\nInjecting chaotic tool to demonstrate graceful failure handling...")
    TOOL_REGISTRY["extract_and_summarize_chunk"] = chaotic_extract_and_summarize_chunk
    
    # 3. Run the loop
    print("\nStarting Agentic Loop...\n")
    run_agentic_loop(pdf_file)
