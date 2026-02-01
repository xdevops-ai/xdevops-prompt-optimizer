# llm_engine/providers/openai.py

import os
from typing import List, Dict, AsyncIterator, Optional, Any
from loguru import logger

# Relative imports to access the base/capabilities from the parent package
from ..base import AbstractLLMClient, LLMResult, LLMTransientError, LLMFatalError
from ..capabilities import is_reasoning_model, build_json_system_instructions

class OpenAIClient(AbstractLLMClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required. Install via 'pip install openai'")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _consolidate_system_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        For O-series (reasoning) models which usually strictly enforce 'user' and 'assistant' roles,
        this merges the system prompt into the first User message.
        """
        system_text = " ".join(m["content"] for m in messages if m["role"] == "system")
        # Filter out system messages
        new_messages = [m.copy() for m in messages if m["role"] != "system"]
        
        if system_text:
            if new_messages and new_messages[0]["role"] == "user":
                # Prepend to the first user message
                existing = new_messages[0]["content"] or ""
                new_messages[0]["content"] = f"System Instruction:\n{system_text}\n\nUser Query:\n{existing}"
            else:
                # If no user message exists or first msg is assistant, insert new user message
                new_messages.insert(0, {"role": "user", "content": system_text})
        
        return new_messages

    def _inject_json_instruction(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Manually injects JSON formatting instructions for models that don't support native JSON mode.
        Returns a new list of messages.
        """
        if not messages:
            return messages
        
        msgs_copy = [m.copy() for m in messages]
        instruction = build_json_system_instructions()
        
        # Append to the very last message to ensure it's fresh in context
        last_msg = msgs_copy[-1]
        last_msg["content"] = (last_msg.get("content") or "") + "\n\n" + instruction
        return msgs_copy

    def _prepare_request_params(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        json_mode: bool, 
        max_output_tokens: Optional[int], 
        temperature: Optional[float], 
        extra_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Centralized logic to normalize parameters for OpenAI's API.
        Handles the split between standard GPT models and O-series (Reasoning) models.
        """
        is_reasoning = is_reasoning_model(model)
        
        # 1. Handle Message Consolidation for Reasoning Models
        if is_reasoning:
            messages = self._consolidate_system_messages(messages)

        # 2. Handle JSON Mode
        if json_mode:
            if is_reasoning:
                # Reasoning models often don't support response_format={"type": "json_object"}
                # So we inject instructions into the prompt text instead.
                messages = self._inject_json_instruction(messages)
            else:
                # Standard models use the native parameter
                extra_kwargs["response_format"] = {"type": "json_object"}

        # 3. Base Parameters
        params = {
            "model": model,
            "messages": messages,
        }

        # 4. Handle Max Tokens (The critical fix)
        # O-series use 'max_completion_tokens', standard use 'max_tokens'
        if max_output_tokens:
            if is_reasoning:
                params["max_completion_tokens"] = max_output_tokens
            else:
                params["max_tokens"] = max_output_tokens

        # 5. Handle Temperature
        # Reasoning models often don't support temperature or force it to 1.0
        if temperature is not None:
            if not is_reasoning:
                params["temperature"] = temperature
            # else: explicitly ignore temperature for reasoning to avoid 400 errors

        # 6. Merge extra kwargs (allowing overrides)
        # We perform a safe update to ensure we don't accidentally send 'max_tokens' to a reasoning model
        # if it was passed in extra_kwargs
        for k, v in extra_kwargs.items():
            if k == "max_tokens" and is_reasoning:
                params["max_completion_tokens"] = v
            elif k == "temperature" and is_reasoning:
                continue
            else:
                params[k] = v

        return params

    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        *, 
        json_mode: bool = False, 
        max_output_tokens: Optional[int] = None, 
        temperature: Optional[float] = None, 
        **kwargs
    ) -> LLMResult:
        import openai

        # Prepare parameters centrally
        params = self._prepare_request_params(
            messages, model, json_mode, max_output_tokens, temperature, kwargs
        )

        try:
            response = await self.client.chat.completions.create(**params)
            choice = response.choices[0]
            
            return LLMResult(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason,
                model=response.model,
                usage=dict(response.usage) if response.usage else {}
            )

        except openai.RateLimitError as e:
            logger.warning(f"OpenAI Rate Limit: {e}")
            raise LLMTransientError(f"Rate limit exceeded: {e}") from e
        except openai.APIConnectionError as e:
            logger.warning(f"OpenAI Connection Error: {e}")
            raise LLMTransientError(f"Connection failed: {e}") from e
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Auth Error: {e}")
            raise LLMFatalError(f"Authentication failed: {e}") from e
        except openai.BadRequestError as e:
            # Catch 400 errors (like invalid params) as Fatal
            logger.error(f"OpenAI Bad Request: {e}")
            raise LLMFatalError(f"Bad request: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected OpenAI Error: {e}")
            raise LLMFatalError(f"Unknown error: {e}") from e

    async def stream(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        *,
        json_mode: bool = False,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        
        # Prepare parameters centrally
        params = self._prepare_request_params(
            messages, model, json_mode, max_output_tokens, temperature, kwargs
        )
        params["stream"] = True

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        yield content
                        
        except Exception as e:
            logger.error(f"OpenAI Stream Error: {e}")
            # For streams, we often yield the error text so the UI sees it immediately
            yield f" [Error: {str(e)}]"

    async def close(self):
        await self.client.close()


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek uses the OpenAI SDK but with a specific Base URL.
    Inherits all the robust param handling from OpenAIClient.
    """
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url="https://api.deepseek.com/v1")