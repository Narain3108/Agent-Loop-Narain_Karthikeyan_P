REASON_PROMPT = """You are the Planning (Reason) module of an agentic loop tasked with progressively summarizing a long document.

You have access to the following tools:
{tools}

CURRENT OBSERVATION (State of the world):
{observation}

MEMORY (Context from previous iterations and semantic recall):
{memory}

Analyze the observation and memory. Determine the next logical step to summarize the document.
- Use `recent_session_history` in Memory to understand what you just did and avoid repeating failed actions.
- Use `relevant_past_summaries` in Memory to maintain context if you need to stitch chunks together.
- If there are unsummarized words/tokens, plan to use `extract_and_summarize_chunk`. Pick a reasonable chunk size (e.g., 40-50 words).
- If the entire document has been summarized, plan to use `compile_final_summary` using the accumulated summaries from the observation.

You must output a JSON conforming to the requested schema, containing:
1. `reasoning_trace`: A step-by-step explanation of your logic.
2. `chosen_action`: The exact name of the tool to call.
3. `parameters`: A dictionary of arguments for the tool.
"""

REFLECT_PROMPT = """You are the Evaluator (Reflect) module of an agentic loop.
Your job is to evaluate the outcome of the last action and determine if the overarching goal is met, or if course correction is needed.

LAST ACTION RESULT:
{result}

CURRENT OBSERVATION (State of the world before the action):
{observation}

Evaluate the result based on these criteria:
1. If the action was `extract_and_summarize_chunk`, look at the `parameters` in the LAST ACTION RESULT to see exactly which word indices were processed. Check if the summary is coherent. Ensure you instruct the Reason step to pick up exactly where this chunk left off (e.g., if end_word_idx was 50, next instruction must say to start at 50). Do NOT claim the entire document is finished unless the `end_word_idx` is equal to or greater than the `total_words` in the observation.
2. If the action was `compile_final_summary`, check if it successfully combined everything.

Determine if the agent is done (i.e., final summary is compiled).
Assign a quality score (0.0 to 1.0).
Provide the `next_instruction` to guide the Reason step in the next iteration. For example: "Chunk 0 to 500 summarized successfully. Proceed to chunk 500 to 1000."

You must output a JSON conforming to the requested schema, containing:
1. `is_done`: boolean
2. `quality_score`: float
3. `next_instruction`: string
"""

SUMMARIZE_CHUNK_PROMPT = """Summarize the following text extracted from a document.
Capture the key points, main arguments, and any critical details.
Keep the summary concise but comprehensive.

TEXT:
{text}

SUMMARY:
"""
