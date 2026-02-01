from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncIterator, Optional
from loguru import logger

@dataclass
class LLMResult:
    """Standardized response object for blocking calls."""
    text: str
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Safe accessor for total tokens."""
        return self.usage.get("total_tokens", 0)

class LLMError(Exception):
    """Base class for all LLM Engine errors."""
    pass

class LLMTransientError(LLMError):
    """
    Retryable errors.
    Examples: 429 Rate Limit, 503 Service Unavailable, Network Timeout.
    Router should catch this and attempt retry with backoff.
    """
    pass

class LLMFatalError(LLMError):
    """
    Non-retryable errors.
    Examples: 401 Unauthorized, 400 Bad Request, Context Length Exceeded.
    Router should catch this and fail immediately.
    """
    pass

class AbstractLLMClient(ABC):
    """
    Abstract contract for AI Providers (OpenAI, Gemini, DeepSeek).
    Enforces consistent behavior for JSON mode, Streaming, and Error handling.
    """

    @abstractmethod
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
        """
        Blocking generation.
        
        Args:
            messages: Standard OpenAI-format message list [{"role": "user", "content": "..."}]
            model: Model identifier string.
            json_mode: If True, provider MUST enforce JSON output (via API param or Prompt Injection).
            max_output_tokens: Limit for generation (handles max_tokens vs max_completion_tokens internally).
            temperature: Randomness (providers may ignore this for Reasoning models).
            **kwargs: Provider-specific extras (top_p, frequency_penalty, etc).

        Returns:
            LLMResult containing the full text and metadata.

        Raises:
            LLMTransientError: If the call can be retried.
            LLMFatalError: If the call failed permanently.
        """
        pass

    @abstractmethod
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
        """
        Streaming generation.

        Yields:
            str: User-visible text chunks. 
            
        Invariant: 
            MUST yield only non-empty strings. 
            Providers must filter keep-alive signals or empty deltas internally.
        """
        pass

    async def close(self) -> None:
        """Optional hook to close underlying HTTP sessions."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()