import os
from loguru import logger
from .base import AbstractLLMClient
# Import from the providers package, ensuring decoupling
from .providers import OpenAIClient, GeminiClient, DeepSeekClient

def get_llm_client() -> AbstractLLMClient:
    """
    Factory function to instantiate the correct LLM provider based on environment config.
    
    Env Vars:
        LLM_PROVIDER: "openai" (default), "gemini", or "deepseek".
        OPENAI_API_KEY: Required for 'openai'.
        GEMINI_API_KEY: Required for 'gemini'.
        DEEPSEEK_API_KEY: Required for 'deepseek'.
        
    Returns:
        An instance of a class implementing AbstractLLMClient.
        
    Raises:
        ValueError: If the required API key for the selected provider is missing.
    """
    # 1. Determine Provider (Default to OpenAI)
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    logger.info(f"Initializing LLM Client for provider: '{provider}'")

    # 2. Instantiate with Validation
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.critical("Missing GEMINI_API_KEY env var for provider 'gemini'.")
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER='gemini'.")
        return GeminiClient(api_key=api_key)

    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.critical("Missing DEEPSEEK_API_KEY env var for provider 'deepseek'.")
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER='deepseek'.")
        return DeepSeekClient(api_key=api_key)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.critical("Missing OPENAI_API_KEY env var for provider 'openai'.")
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER='openai'.")
        return OpenAIClient(api_key=api_key)

    else:
        # Fallback / Error for unknown providers
        logger.warning(f"Unknown LLM_PROVIDER '{provider}'. Defaulting to OpenAI.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"Unknown provider '{provider}' and fallback OPENAI_API_KEY is missing.")
        return OpenAIClient(api_key=api_key)