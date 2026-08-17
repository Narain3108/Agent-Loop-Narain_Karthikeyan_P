import os
from typing import Dict, Any, Callable
from pypdf import PdfReader
from google import genai
from google.genai import types
from agent.prompts import SUMMARIZE_CHUNK_PROMPT

# Tool definitions in JSON Schema format
TOOL_DEFINITIONS = [
    {
        "name": "extract_and_summarize_chunk",
        "description": "Extracts a specific chunk of text based on word indices and summarizes it using the LLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The path to the PDF document."
                },
                "start_word_idx": {
                    "type": "integer",
                    "description": "The starting word index."
                },
                "end_word_idx": {
                    "type": "integer",
                    "description": "The ending word index (exclusive)."
                }
            },
            "required": ["filepath", "start_word_idx", "end_word_idx"]
        }
    },
    {
        "name": "compile_final_summary",
        "description": "Compiles all the individual chunk summaries into a final coherent summary. Use this only when all chunks have been summarized.",
        "parameters": {
            "type": "object",
            "properties": {
                "summaries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of summaries generated from the document chunks."
                }
            },
            "required": ["summaries"]
        }
    }
]

def _extract_all_text(filepath: str) -> str:
    """Helper function to extract all text from a PDF."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

def extract_and_summarize_chunk(filepath: str, start_word_idx: int, end_word_idx: int) -> str:
    """
    Reads the document, extracts the word chunk, and summarizes it via LLM.
    """
    try:
        full_text = _extract_all_text(filepath)
        words = full_text.split()
        chunk_words = words[start_word_idx:end_word_idx]
        chunk_text = " ".join(chunk_words)
        
        if not chunk_text.strip():
            return "Error: Extracted chunk is empty."

        # Initialize the GenAI client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set."
            
        client = genai.Client(api_key=api_key)
        
        prompt = SUMMARIZE_CHUNK_PROMPT.format(text=chunk_text)
        
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        
        return response.text
    except Exception as e:
        return f"Error extracting/summarizing chunk: {str(e)}"

def compile_final_summary(summaries: list[str]) -> str:
    """
    Concatenates the summaries into a final coherent string.
    """
    combined = "\n\n--- Next Section ---\n\n".join(summaries)
    return f"FINAL COMPILED SUMMARY:\n\n{combined}"

# Tool registry to map function names to actual callables
TOOL_REGISTRY: Dict[str, Callable] = {
    "extract_and_summarize_chunk": extract_and_summarize_chunk,
    "compile_final_summary": compile_final_summary,
}
