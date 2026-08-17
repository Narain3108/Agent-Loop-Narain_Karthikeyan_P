import time
import random
import yaml
from functools import wraps
from typing import Callable, Any
from google.genai.errors import APIError

def load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback default config if missing
        return {
            "llm": {"model": "gemini-3.5-flash", "temperature": 0.2},
            "loop": {"max_iterations": 10, "infinite_loop_threshold": 2},
            "harness": {"max_retries": 3, "base_delay": 1.0, "max_delay": 10.0, "token_budget": 100000}
        }

CONFIG = load_config()

def with_retry(func: Callable) -> Callable:
    """
    Decorator for exponential backoff with jitter.
    Catches network API errors and JSONDecodeError (unparseable output).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = CONFIG["harness"]["max_retries"]
        base_delay = CONFIG["harness"]["base_delay"]
        max_delay = CONFIG["harness"]["max_delay"]
        
        retries = 0
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # If we've hit max retries, propagate the error (or fallback in orchestrator)
                if retries == max_retries:
                    print(f"[HARNESS] Max retries reached for {func.__name__}. Error: {e}")
                    raise e
                
                # Calculate delay with jitter
                delay = min(max_delay, base_delay * (2 ** retries))
                jitter = random.uniform(0, 0.1 * delay)
                sleep_time = delay + jitter
                
                print(f"[HARNESS] Retry {retries + 1}/{max_retries} for {func.__name__} in {sleep_time:.2f}s due to: {type(e).__name__}({e})")
                
                # Check if it's a JSON unparseable error during LLM generation
                # In that case, we can inject a simplified prompt fallback here or let the orchestrator handle it.
                # For now, we will simply wait and retry, hoping temperature variations fix it.
                time.sleep(sleep_time)
                retries += 1
    return wrapper
