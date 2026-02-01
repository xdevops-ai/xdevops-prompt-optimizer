import json
import logging
import sys
import os
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Config first to ensure .env is loaded before llm_engine initializes
from config import Config
from llm_engine.factory import get_llm_client
from llm_engine.base import LLMTransientError, LLMFatalError

# Configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMClient")

class AsyncLLMClient:
    def __init__(self):
        self.client = get_llm_client()

    async def generate_response(
        self, 
        system_prompt: str, 
        user_message: str, 
        model: str = Config.MODEL_SMART,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generates a JSON response from the LLM.
        Delegates to llm_engine for model normalization (O-series vs GPT-4).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            # llm_engine handles:
            # 1. Consolidating system messages for reasoning models
            # 2. Mapping max_output_tokens to max_tokens vs max_completion_tokens
            # 3. Ignoring temperature for reasoning models
            result = await self.client.generate(
                messages=messages,
                model=model,
                json_mode=True,
                temperature=temperature,
                # We omit max_output_tokens to let the model/API use its default limits.
                # This ensures compatibility with future models (e.g. o3, gpt-5) without code changes.
            )
            
            if result.finish_reason == "length":
                logger.error("LLM response truncated (finish_reason='length').")
                return {
                    "error": "Response truncated. Output limit exceeded.",
                    "raw_content": result.text
                }

            if not result.text:
                return {"error": "Empty response from LLM."}

            return json.loads(result.text)

        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON output from LLM."}
        except LLMTransientError as e:
            logger.warning(f"Transient LLM Error: {e}")
            return {"error": f"Transient Error: {str(e)}"}
        except LLMFatalError as e:
            logger.error(f"Fatal LLM Error: {e}")
            return {"error": f"Fatal Error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return {"error": f"Unexpected Error: {str(e)}"}