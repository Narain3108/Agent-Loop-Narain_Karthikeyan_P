# Memory Architecture (Milestone 2)

This agentic loop implements a **Dual-Memory System** to maintain both immediate execution context and long-term semantic understanding without relying on any external agent frameworks (like LangChain or LlamaIndex).

## 1. Short-Term Session Memory (Ephemeral List)
**Tool:** Native Python List (`self.session_mem`)
**Purpose:** Provides the agent with immediate context of its most recent actions to prevent infinite loops and repeat errors.
**Implementation:**
- Stores a rolling window of the last 3 iterations.
- Each entry contains the full iteration trace: `observation`, `plan` (from reason), `action_result` (from act), and `reflection` (from reflect).

**Concrete Example in Action:**
If the agent attempts to read word chunk `0 to 1000` and the action fails (e.g., PDF read error), the `reflect` stage will catch this and issue a low score. In iteration N+1, the agent reads its `recent_session_history`, sees that chunk `0 to 1000` just failed, and will reason: *"I see my last attempt to read 0-1000 failed. I will try reading a smaller chunk 0-500 instead to see if that resolves the error."*

## 2. Long-Term Semantic Memory (ChromaDB)
**Tool:** `chromadb` (Raw Python Package)
**Purpose:** Provides the agent with semantic recall of past summaries, preventing it from losing context over very long documents where early summaries would otherwise fall out of the context window.
**Implementation:**
- Runs entirely in-memory using `chromadb.EphemeralClient()` as requested.
- When `extract_and_summarize_chunk` is successful, the generated summary text is stored as a document in the Chroma collection. Chroma automatically generates vector embeddings for it.
- During the `reason` stage, the agent's current task instruction is used as a query to perform a semantic search against the database. The most relevant past summaries are injected into the prompt.

**Concrete Example in Action:**
If the agent is summarizing chunk 10 (words 10,000 to 11,000) and encounters a pronoun like "He", it might not know who "He" is. The memory manager queries ChromaDB with the current context, retrieves the summary from chunk 2 which explains who the character is, and injects it into `relevant_past_summaries`. The Reason stage uses this to stitch the narrative together accurately.

## 3. Data Clearing Lifecycle
To ensure the session stays clean across different documents, `main.py` explicitly calls `memory_manager.clear()` once the `is_done` flag is thrown by the Reflect stage. This empties the Python list and entirely deletes the ephemeral Chroma collection.
