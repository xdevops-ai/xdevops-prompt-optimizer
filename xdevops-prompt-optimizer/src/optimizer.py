import json
from typing import List, Dict
from llm_client import AsyncLLMClient
from templates import MetaPrompts
from config import Config

class Architect:
    def __init__(self, llm_client: AsyncLLMClient):
        self.client = llm_client

    async def repair_prompt(self, current_prompt: str, failure_logs: List[Dict]) -> str:
        """
        Analyzes failure logs and requests a repaired system prompt using the file-based meta-prompt.
        """
        # 1. Load the Meta-Prompt Dynamically
        try:
            system_instruction = MetaPrompts.load_architect_system()
        except Exception as e:
            print(f"Architect Critical Error: Could not load meta-prompt. {e}")
            return current_prompt

        # 2. Format failures
        failure_text = self._format_failures(failure_logs)
        
        # 3. Construct the User Message
        user_message = MetaPrompts.ARCHITECT_USER_TEMPLATE.format(
            current_prompt=current_prompt,
            failures=failure_text
        )

        # 4. Call the LLM
        response = await self.client.generate_response(
            system_prompt=system_instruction,  # Passing the loaded file content
            user_message=user_message,
            model=Config.MODEL_SMART,
            temperature=Config.TEMPERATURE_ARCHITECT
        )

        # 5. Extract Result (Direct JSON Dump)
        if "error" in response:
            print(f"Architect Error: {response['error']}")
            return current_prompt

        try:
            if isinstance(response, dict):
                 return json.dumps(response, indent=2)
            else:
                 return str(response)
        except Exception as e:
            print(f"Error parsing Architect response: {e}")
            return current_prompt

    def _format_failures(self, logs: List[Dict]) -> str:
        formatted = []
        for i, log in enumerate(logs):
            if i >= 10: 
                formatted.append("... (more failures truncated) ...")
                break
            
            # Sanitize inputs to prevent log injection confusion
            safe_input = str(log.get('input', 'N/A')).replace('\n', ' ')
            safe_error = str(log.get('error_message', 'Unknown')).replace('\n', ' ')
            
            entry = (
                f"- Failure #{i+1}: "
                f"Input='{safe_input}', "
                f"Error='{safe_error}', "
                f"Actual Output={json.dumps(log.get('actual', {}))}"
            )
            formatted.append(entry)
        
        return "\n".join(formatted)
    

class EfficiencyExpert:
    def __init__(self, llm_client: AsyncLLMClient):
        self.client = llm_client

    async def optimize_prompt(self, current_prompt: str) -> str:
        """
        Requests a structural efficiency pass on the system prompt.
        """
        # 1. Load Efficiency Meta-Prompt
        try:
            system_instruction = MetaPrompts.load_efficiency_system()
        except Exception as e:
            print(f"EfficiencyExpert Critical Error: {e}")
            return current_prompt

        # 2. Construct User Message
        user_message = MetaPrompts.EFFICIENCY_USER_TEMPLATE.format(
            current_prompt=current_prompt
        )

        # 3. Call LLM 
        response = await self.client.generate_response(
            system_prompt=system_instruction,
            user_message=user_message,
            model=Config.MODEL_SMART,
            temperature=Config.TEMPERATURE_EFFICIENCY 
        )

        # 4. Extract Result
        if "error" in response:
            print(f"EfficiencyExpert Error: {response['error']}")
            return current_prompt

        try:
            if isinstance(response, dict):
                 return json.dumps(response, indent=2)
            else:
                 return str(response)
        except Exception as e:
            print(f"Error parsing EfficiencyExpert response: {e}")
            return current_prompt