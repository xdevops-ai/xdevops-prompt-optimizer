from typing import List, Dict, AsyncIterator, Optional, Any
from loguru import logger
from ..base import AbstractLLMClient, LLMResult, LLMFatalError

class GeminiClient(AbstractLLMClient):
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai is required. Install via 'pip install google-generativeai'")
        
        genai.configure(api_key=api_key)
        self.genai = genai

    async def generate(self, messages: List[Dict[str, str]], model: str, *, json_mode: bool = False, max_output_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> LLMResult:
        gemini_model = self.genai.GenerativeModel(model)
        chat = self._convert_history(messages)
        
        config = {}
        if max_output_tokens: config["max_output_tokens"] = max_output_tokens
        if temperature is not None: config["temperature"] = temperature
        if json_mode: config["response_mime_type"] = "application/json"

        try:
            response = await gemini_model.generate_content_async(chat, generation_config=config)
            
            finish_reason = "stop"
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = str(response.candidates[0].finish_reason.name)

            return LLMResult(
                text=response.text,
                finish_reason=finish_reason,
                model=model,
                usage={}
            )
        except Exception as e:
            logger.error(f"Gemini Generate Error: {e}")
            raise LLMFatalError(str(e))

    async def stream(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[str]:
        gemini_model = self.genai.GenerativeModel(model)
        chat = self._convert_history(messages)
        
        try:
            config = {}
            if "max_output_tokens" in kwargs:
                config["max_output_tokens"] = kwargs["max_output_tokens"]

            response_stream = await gemini_model.generate_content_async(
                chat, 
                stream=True,
                generation_config=config if config else None
            )
            
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini Stream Error: {e}")
            raise LLMFatalError(str(e))

    def _convert_history(self, messages: List[Dict[str, str]]) -> str:
        # Simple string conversion
        prompt = ""
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            prompt += f"{role}: {content}\n"
        return prompt