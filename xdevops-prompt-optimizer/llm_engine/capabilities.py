import os
import math
import tiktoken
from typing import Set, Dict
from loguru import logger

# --- Configuration Constants ---

# Hardcoded prefixes for known reasoning models (o1, o3, etc.)
# Matches logic in bot.py REASONING_MODELS + expansion for o4/gpt-5
DEFAULT_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

# Context Windows (Source of Truth extracted from bot.py)
# Used to calculate truncation limits and reply caps.
DEFAULT_CONTEXT_WINDOWS = {
    "o4-mini": 200_000,
    "o3-mini": 128_000,
    "o1-mini": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    # Fallback for unknown models
    "default": 128_000
}

# Heuristic divisor for non-OpenAI models (conservative estimate)
TOKEN_ESTIMATION_DIVISOR = 3.5

def _get_reasoning_overrides() -> Set[str]:
    """
    Parses the REASONING_MODEL_OVERRIDES environment variable.
    """
    raw = os.getenv("REASONING_MODEL_OVERRIDES", "")
    if not raw:
        return set()
    return {m.strip().lower() for m in raw.split(",") if m.strip()}

def is_reasoning_model(model: str) -> bool:
    """
    Determines if a model requires 'max_completion_tokens' (Reasoning) 
    instead of 'max_tokens' and NO 'response_format'.
    """
    if not model: return False
    model_lower = model.lower().strip()
    
    # 1. Check overrides
    overrides = _get_reasoning_overrides()
    if model_lower in overrides:
        return True
        
    # 2. Check standard prefixes
    return any(model_lower.startswith(p) for p in DEFAULT_REASONING_PREFIXES)

def json_mode_strategy(model: str) -> str:
    """
    Returns 'api' (response_format) or 'prompt' (injection) strategy.
    """
    if is_reasoning_model(model):
        return "prompt"
    return "api"

def get_context_window(model: str) -> int:
    """
    Returns the total context window size for a model.
    """
    if not model: 
        return DEFAULT_CONTEXT_WINDOWS["default"]
    
    model_lower = model.lower().strip()
    return DEFAULT_CONTEXT_WINDOWS.get(model_lower, DEFAULT_CONTEXT_WINDOWS["default"])

def compute_max_output_tokens(model: str) -> int:
    """
    Calculates the dynamic output token limit based on context size.
    Logic strictly mirrors bot.py: _refine_scope_gen -> compute_reply_cap
    """
    ctx_limit = get_context_window(model)
    
    # Defaults match bot.py
    pct = float(os.getenv("REPLY_CAP_PCT", "0.08")) 
    cap_min = int(os.getenv("REPLY_CAP_MIN", "1200"))
    cap_max = int(os.getenv("REPLY_CAP_MAX", "6000"))
    
    calculated = int(ctx_limit * pct)
    return max(cap_min, min(cap_max, calculated))

def estimate_tokens(text: str, model: str) -> int:
    """
    Centralized token budget estimation.
    Safe for production: catches errors and falls back gracefully.
    """
    if not text:
        return 0
    
    model_lower = model.lower()
    
    # Attempt Tiktoken (OpenAI Standard)
    # Checks for gpt, o1, o3, o4 to try specific encoding
    if any(k in model_lower for k in ("gpt", "o1", "o3", "o4")):
        try:
            enc = tiktoken.encoding_for_model(model)
            # disallowed_special=() prevents crashes on special tokens like <|endoftext|>
            return len(enc.encode(text, disallowed_special=()))
        except KeyError:
            # Model unknown to this version of tiktoken, try generic fallback
            pass
        except Exception as e:
            logger.warning(f"Tiktoken error for model '{model}': {e}")

        # Fallback to cl100k_base (standard for GPT-4/3.5/o1)
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    
    # Ultimate Fallback: Conservative Heuristic
    return math.ceil(len(text) / TOKEN_ESTIMATION_DIVISOR)

def build_json_system_instructions() -> str:
    """
    Returns the standardized system prompt injection for enforcing JSON.
    """
    return (
        "\nIMPORTANT: Return ONLY a valid JSON object. "
        "Do not include markdown formatting like ```json. "
        "Do not include any explanation, preamble, or postscript."
    )