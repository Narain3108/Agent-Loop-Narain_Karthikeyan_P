# Agentic Patterns Research

In modern agentic systems, several patterns dictate how an LLM handles reasoning and action. Below is a brief overview of the required patterns:

- **ReAct (Reason + Act)**: The LLM interleaves reasoning traces with actions. It thinks about what to do, takes an action (using a tool), observes the result, and repeats. This is excellent for multi-step tasks where the next step depends on the outcome of the previous one.
- **Reflexion**: A self-evaluation loop with verbal reinforcement. The agent takes an action, evaluates the quality or success of the result (reflects), and uses this reflection as context in the next iteration to correct mistakes or hallucination.
- **Chain-of-Thought (CoT)**: The model is prompted to output step-by-step reasoning before arriving at a final answer. This forces the LLM to allocate computational tokens to planning and logic, improving accuracy on complex tasks.
- **Tree of Thoughts (ToT)**: Explores multiple reasoning branches simultaneously. It generates several possible next steps, evaluates them, and searches through the tree (e.g., using BFS or DFS) to find the optimal path.
- **LATS (Language Agent Tree Search)**: Combines Monte Carlo Tree Search (MCTS) with LLM reasoning. It samples actions, evaluates states, and backpropagates values to build a search tree, balancing exploration and exploitation.

## Chosen Pattern: Reflexion + CoT
For this progressive document summarization agent, we have chosen a combination of **Reflexion** and **Chain-of-Thought (CoT)**.

**Why this fits the use case:**
Progressive summarization requires high accuracy and completeness without hallucinating facts not present in the document chunks. 
- We use **CoT** during the `Reason` stage (enforced via the `reasoning_trace` parameter) so the agent can methodically plan which word chunks (start/end indices) to extract and summarize next.
- We strictly implement the **Reflexion** pattern in our `Reflect` stage. After the `Act` stage generates a summary using `extract_and_summarize_chunk`, the LLM evaluates the summary against the observation. It assigns a quality score and generates the next instruction. If the summary is poor or hallucinated, Reflexion allows the loop to reject the result and instruct the Reason stage to try again. If it is good, it reinforces the loop to proceed to the next chunk.
