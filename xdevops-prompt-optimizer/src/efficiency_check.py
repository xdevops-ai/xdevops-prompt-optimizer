import asyncio
import os
from llm_client import AsyncLLMClient
from optimizer import EfficiencyExpert
from config import Config
from metrics import MetricsEngine

DUMMY_PROMPT = """
{
  "title": "Verbose Prompt",
  "instructions": "You are a very helpful AI. You must always listen to the user. You must output JSON. The JSON must be valid. Please do not output markdown.",
  "rules": [
    "Rule 1: Use equality operators.",
    "Rule 2: Use equality operators like =="
  ]
}
"""

async def run_efficiency_test():
    print("--- Starting Efficiency Expert Test ---")
    
    if not Config.OPENAI_API_KEY:
        print("FAIL: No API Key found.")
        return
        
    if not os.path.exists(Config.META_PROMPT_EFFICIENCY_PATH):
        print(f"FAIL: {Config.META_PROMPT_EFFICIENCY_PATH} not found.")
        return

    client = AsyncLLMClient()
    expert = EfficiencyExpert(client)
    metrics = MetricsEngine()
    
    original_tokens = metrics.count_tokens(DUMMY_PROMPT)
    print(f"Original Tokens: {original_tokens}")
    
    print("Requesting Optimization...")
    new_prompt = await expert.optimize_prompt(DUMMY_PROMPT)
    
    new_tokens = metrics.count_tokens(new_prompt)
    print(f"New Tokens: {new_tokens}")
    
    print("\n--- Result ---")
    print(new_prompt)
    
    if new_tokens < original_tokens:
        print(f"\nSUCCESS: EfficiencyExpert reduced tokens by {original_tokens - new_tokens}.")
    else:
        print("\nWARNING: Tokens not reduced. The prompt might already be dense.")

if __name__ == "__main__":
    asyncio.run(run_efficiency_test())