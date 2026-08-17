import os
import json
import time
from dotenv import load_dotenv

from agent.loop import perceive, reason, act, reflect
from agent.tools import TOOL_REGISTRY

# Load environment variables (API Key)
load_dotenv()

def run_agentic_loop(filepath: str, max_iterations: int = 10):
    print(f"Starting Agentic Loop for {filepath}")
    
    # Initialize the input data for the first perceive call
    input_data = json.dumps({"filepath": filepath})
    
    observation = {}
    memory = [] # Milestone 1: memory is empty
    
    for i in range(max_iterations):
        print(f"\n========== ITERATION {i+1} ==========")
        
        # 1. PERCEIVE
        observation = perceive(input_data)
        if "error" in observation:
            print(f"Perceive Error: {observation['error']}")
            break
            
        # 2. REASON
        plan = reason(observation, memory)
        if "error" in plan:
            print(f"Reason Error: {plan['error']}")
            break
            
        # 3. ACT
        result = act(plan, TOOL_REGISTRY)
        if result.get("status") == "error":
            print(f"Act Error: {result['message']}")
            # In a robust harness, we might retry, but for M1 we can just pass the error to reflect
            
        # 4. REFLECT
        reflection = reflect(result, observation)
        
        # Loop state management
        if reflection.get("is_done"):
            print("\n>>> GOAL ACCOMPLISHED! <<<")
            if result.get("action") == "compile_final_summary":
                print(f"Final Result:\n{result.get('result_content')}")
            break
            
        # If the action was successful and it was an extraction, we should save the summary
        if reflection.get("quality_score", 0) >= 0.7:
            if result.get("action") == "extract_and_summarize_chunk":
                summary = result.get("result_content")
                observation["summaries_collected"].append(summary)
                print(f"-> Summary appended. Total summaries: {len(observation['summaries_collected'])}")
        
        # Feed the reflection and current observation back into the next perceive step
        input_data = json.dumps({
            "filepath": filepath,
            "previous_observation": observation,
            "reflection": reflection
        })
        
        time.sleep(2) # Small delay to avoid rate limiting
        
    else:
        print("\n>>> MAX ITERATIONS REACHED <<<")

if __name__ == "__main__":
    test_filepath = "sample.pdf"
    
    if not os.path.exists(test_filepath):
        print(f"Please create a '{test_filepath}' file in this directory to run the demo.")
    else:
        run_agentic_loop(test_filepath, max_iterations=5)
