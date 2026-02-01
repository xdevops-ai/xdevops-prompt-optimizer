import os
from config import Config

class MetaPrompts:
    
    ARCHITECT_USER_TEMPLATE = """
    The current system prompt:
    {current_prompt}
    
    A summary of testing failures:
    {failures}
    """

    EFFICIENCY_USER_TEMPLATE = """
    The current system prompt (100% Functional):
    {current_prompt}
    
    INSTRUCTIONS:
    Apply your optimization strategies to the prompt above.
    Goal: Reduce token count without altering logic.
    """

    @staticmethod
    def _load_file(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Meta-prompt file not found at: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_architect_system() -> str:
        return MetaPrompts._load_file(Config.META_PROMPT_PATH)

    @staticmethod
    def load_efficiency_system() -> str:
        return MetaPrompts._load_file(Config.META_PROMPT_EFFICIENCY_PATH)