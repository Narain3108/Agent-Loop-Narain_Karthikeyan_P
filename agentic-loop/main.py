import os
import json
import time
from dotenv import load_dotenv

from agent.loop import perceive, reason, act, reflect
from agent.tools import TOOL_REGISTRY
from agent.memory_manager import MemoryManager
from agent.harness import CONFIG
from agent.logger import StructuredLogger

# Load environment variables (API Key)
load_dotenv()

def run_agentic_loop(filepath: str):
    max_iterations = CONFIG["loop"]["max_iterations"]
    infinite_loop_threshold = CONFIG["loop"]["infinite_loop_threshold"]
    token_budget = CONFIG["harness"]["token_budget"]
    
    print(f"Starting Agentic Loop for {filepath} with max {max_iterations} iterations")
    
    logger = StructuredLogger()
    input_data = json.dumps({"filepath": filepath})
    observation = {}
    memory = MemoryManager()
    
    cumulative_tokens = 0
    consecutive_same_plan = 0
    last_plan_str = ""
    best_result = None
    
    for i in range(max_iterations):
        start_time = time.time()
        print(f"\n========== ITERATION {i+1} ==========")
        
        # 1. PERCEIVE
        perc_start = time.time()
        observation = perceive(input_data)
        perc_lat = (time.time() - perc_start) * 1000
        
        if "error" in observation:
            print(f"Perceive Error: {observation['error']}")
            logger.log_step(i+1, "perceive", observation, latency_ms=perc_lat, error=observation["error"])
            break
        logger.log_step(i+1, "perceive", observation, latency_ms=perc_lat)
            
        # 2. REASON
        reas_start = time.time()
        try:
            recalled_context = memory.recall(query=observation.get("last_instruction"))
        except Exception as e:
            print(f"[HARNESS] Memory read failure: {e}. Proceeding with empty context.")
            recalled_context = {"recent_session_history": [], "relevant_past_summaries": []}
            
        plan = reason(observation, recalled_context)
        reas_lat = (time.time() - reas_start) * 1000
        
        # Track Tokens
        tokens = plan.pop("__tokens__", 0)
        cumulative_tokens += tokens
        
        if "error" in plan:
            print(f"Reason Error: {plan['error']}")
            logger.log_step(i+1, "reason", plan, latency_ms=reas_lat, error=plan["error"])
            break
        logger.log_step(i+1, "reason", plan, latency_ms=reas_lat)
            
        # Infinite Loop Detection
        current_plan_str = json.dumps(plan, sort_keys=True)
        if current_plan_str == last_plan_str:
            consecutive_same_plan += 1
        else:
            consecutive_same_plan = 0
        last_plan_str = current_plan_str
        
        if consecutive_same_plan >= infinite_loop_threshold:
            print(f"[HARNESS] Infinite loop detected! Plan repeated {infinite_loop_threshold} times. Breaking out.")
            break
            
        # 3. ACT
        act_start = time.time()
        result = act(plan, TOOL_REGISTRY)
        act_latency = (time.time() - act_start) * 1000
        
        if result.get("status") == "error":
            print(f"Act Error: {result['message']}")
            logger.log_step(i+1, "act", result, latency_ms=act_latency, error=result['message'])
            # Graceful error: let reflect handle the failure, don't crash
        else:
            logger.log_step(i+1, "act", result, latency_ms=act_latency)
            best_result = result
            
        # 4. REFLECT
        reflect_start = time.time()
        reflection = reflect(result, observation)
        reflect_latency = (time.time() - reflect_start) * 1000
        
        # Track Tokens
        r_tokens = reflection.pop("__tokens__", 0)
        cumulative_tokens += r_tokens
        logger.log_step(i+1, "reflect", reflection, latency_ms=reflect_latency)
        
        # Token Budget Warning
        if cumulative_tokens > token_budget:
            print(f"[HARNESS WARNING] Cumulative tokens ({cumulative_tokens}) exceeded budget ({token_budget})!")
        
        # Save to memory manager (session trace and vector DB)
        summary_to_save = None
        if result.get("action") == "extract_and_summarize_chunk" and reflection.get("quality_score", 0) >= 0.7:
            summary_to_save = result.get("result_content")
            
        memory.save(
            iteration_data={
                "observation": observation,
                "plan": plan,
                "action_result": result,
                "reflection": reflection
            },
            summary_text=summary_to_save
        )
        
        # Loop state management
        if reflection.get("is_done"):
            print("\n>>> GOAL ACCOMPLISHED! <<<")
            if result.get("action") == "compile_final_summary":
                print(f"Final Result:\n{result.get('result_content')}")
            
            # Wipe memory for next session as requested
            memory.clear()
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
        
        loop_latency = (time.time() - start_time) * 1000
        print(f"Iteration completed in {loop_latency:.1f}ms. Total tokens so far: {cumulative_tokens}")
        
    else:
        print("\n>>> MAX ITERATIONS REACHED <<<")
        print("Status: PARTIAL")
        if best_result:
            print(f"Best Partial Result:\n{best_result.get('result_content')}")

if __name__ == "__main__":
    test_filepath = "sample.pdf"
    
    if not os.path.exists(test_filepath):
        print(f"Please create a '{test_filepath}' file in this directory to run the demo.")
    else:
        run_agentic_loop(test_filepath)
