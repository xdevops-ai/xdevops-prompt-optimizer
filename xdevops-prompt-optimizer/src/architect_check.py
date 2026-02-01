import asyncio
import os
from llm_client import AsyncLLMClient
from optimizer import Architect
from config import Config

# Dummy failure to test the Architect
DUMMY_FAILURE = [{
    "input": "Find tasks for Alice",
    "expected": {"command": "search", "filter": "user_id == 'alice'"},
    "actual": {"command": "search", "filter": "text == 'alice'"},
    "error_message": "Metadata filter mismatch: expected user_id check."
}]

DUMMY_PROMPT = """
You are a helper.
When asked for tasks, search by text.
"""

async def run_architect_test():
    print("--- Starting Architect Integration Test ---")
    
    if not Config.OPENAI_API_KEY:
        print("FAIL: No API Key found. Skipping test.")
        return

    print("[1/2] Initializing Client...")
    client = AsyncLLMClient()
    architect = Architect(client)
    
    print("[2/2] Requesting Repair for Dummy Failure...")
    print(f"Original Prompt Length: {len(DUMMY_PROMPT)}")
    
    new_prompt = await architect.repair_prompt(DUMMY_PROMPT, DUMMY_FAILURE)
    
    print("\n--- Repair Result ---")
    print(f"New Prompt Length: {len(new_prompt)}")
    print("Preview of New Prompt:")
    print(new_prompt[:200] + "...")
    
    if len(new_prompt) > len(DUMMY_PROMPT) and "user_id" in new_prompt:
        print("\nSUCCESS: Architect successfully rewrote the prompt and added the missing logic.")
    else:
        print("\nWARNING: Architect ran, but the output may not be improved. Check logs.")

if __name__ == "__main__":
    asyncio.run(run_architect_test())