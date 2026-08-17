import os
import json
from typing import Dict, Any, List, Callable
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from agent.prompts import REASON_PROMPT, REFLECT_PROMPT
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY, _extract_all_text
from agent.harness import with_retry, CONFIG

def _parse_json(text: str) -> Dict[str, Any]:
    """Robust JSON parsing that strips markdown code blocks commonly returned by LLMs."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

# Define Schemas for LLM Structured Output
class ReasonPlan(BaseModel):
    reasoning_trace: str = Field(description="A step-by-step explanation of your logic.")
    chosen_action: str = Field(description="The exact name of the tool to call.")
    parameters: Dict[str, Any] = Field(description="A dictionary of arguments for the tool.")

class Reflection(BaseModel):
    is_done: bool = Field(description="Whether the overarching goal is complete.")
    quality_score: float = Field(description="Score between 0.0 and 1.0 indicating quality of the action result.")
    next_instruction: str = Field(description="Instruction for the next reason step.")


def get_llm_client() -> genai.Client:
    """Helper to initialize the GenAI client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def perceive(input_data: str) -> Dict[str, Any]:
    """
    Parse and structure raw input.
    Extract intent, constraints, context, and relevant signals.
    Returns a structured observation dict.
    
    input_data is expected to be a JSON string containing at minimum a 'filepath',
    and optionally a 'previous_observation' and 'reflection' to maintain loop state.
    """
    print(f"--- PERCEIVE ---")
    try:
        data = json.loads(input_data)
    except json.JSONDecodeError:
        return {"error": "input_data must be a valid JSON string."}
        
    filepath = data.get("filepath", "").strip()
    reflection = data.get("reflection", {})
    prev_obs = data.get("previous_observation", {})
    
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    try:
        full_text = _extract_all_text(filepath)
        words = full_text.split()
        total_words = len(words)
        
        # If this is the first iteration, initialize observation
        if not prev_obs:
            observation = {
                "document_filepath": filepath,
                "total_words": total_words,
                "summaries_collected": [],
                "last_instruction": "Start summarizing the document from word index 0."
            }
        else:
            observation = prev_obs
            # Update observation with reflection feedback
            if reflection:
                observation["last_instruction"] = reflection.get("next_instruction", "")
                
        print(f"Perceived document with {total_words} words. Next instruction: {observation['last_instruction']}")
        return observation
    except Exception as e:
        return {"error": str(e)}


@with_retry
def reason(observation: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the LLM to decide what to do next based on the observation.
    Returns a plan dict: chosen action, parameters, reasoning trace.
    """
    print(f"--- REASON ---")
    client = get_llm_client()
    
    prompt = REASON_PROMPT.format(
        tools=json.dumps(TOOL_DEFINITIONS, indent=2),
        observation=json.dumps(observation, indent=2),
        memory=json.dumps(memory, indent=2)
    )
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=CONFIG["llm"]["temperature"]
    )
    
    response = client.models.generate_content(
        model=CONFIG["llm"]["model"],
        contents=prompt,
        config=config
    )
    
    try:
        # Pydantic models when used as response_schema return JSON text, we parse it safely
        plan_dict = _parse_json(response.text)
        
        # Track tokens
        if getattr(response, "usage_metadata", None):
            plan_dict["__tokens__"] = response.usage_metadata.total_token_count
            
        print(f"Plan: {plan_dict['chosen_action']}")
        print(f"Reasoning: {plan_dict['reasoning_trace']}")
        return plan_dict
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse LLM response into plan",
            "raw_response": response.text
        }


def act(plan: Dict[str, Any], tools: Dict[str, Callable]) -> Dict[str, Any]:
    """
    Execute the planned action by calling the appropriate tool.
    Returns the result of the action.
    """
    print(f"--- ACT ---")
    if "error" in plan:
        return {"status": "error", "message": plan["error"]}
        
    action_name = plan.get("chosen_action")
    parameters = plan.get("parameters", {})
    
    if not action_name or action_name not in tools:
        return {"status": "error", "message": f"Tool '{action_name}' not found or invalid."}
        
    tool_func = tools[action_name]
    print(f"Executing tool: {action_name} with params: {parameters}")
    
    try:
        # Use dictionary unpacking to pass parameters
        result_content = tool_func(**parameters)
        return {
            "status": "success",
            "action": action_name,
            "parameters": parameters,
            "result_content": result_content
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}"
        }


@with_retry
def reflect(result: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate whether the goal was met.
    Return a reflection dict: is_done flag, quality score, next instruction.
    """
    print(f"--- REFLECT ---")
    client = get_llm_client()
    
    prompt = REFLECT_PROMPT.format(
        result=json.dumps(result, indent=2),
        observation=json.dumps(observation, indent=2)
    )
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=CONFIG["llm"]["temperature"]
    )
    
    response = client.models.generate_content(
        model=CONFIG["llm"]["model"],
        contents=prompt,
        config=config
    )
    
    try:
        reflection_dict = _parse_json(response.text)
        
        if getattr(response, "usage_metadata", None):
            reflection_dict["__tokens__"] = response.usage_metadata.total_token_count
            
        print(f"Reflection: is_done={reflection_dict['is_done']}, score={reflection_dict['quality_score']}")
        print(f"Next instruction: {reflection_dict['next_instruction']}")
        return reflection_dict
    except json.JSONDecodeError as e:
        # Fallback if parsing fails
        return {
            "is_done": False,
            "quality_score": 0.0,
            "next_instruction": "Failed to parse reflection, please retry the last step.",
            "error": "Failed to parse LLM response into reflection"
        }
