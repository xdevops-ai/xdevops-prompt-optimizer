import time
from dataclasses import dataclass
from llm_engine.capabilities import estimate_tokens
from config import Config

@dataclass
class RunMetrics:
    accuracy: float         # 0.0 to 1.0
    token_count: int
    latency_ms: float
    pareto_score: float

class MetricsEngine:
    def __init__(self, model_name: str = Config.MODEL_SMART):
        self.model_name = model_name
        self.alpha = Config.ALPHA_ACCURACY
        self.beta = Config.BETA_TOKEN_PENALTY

    def count_tokens(self, text: str) -> int:
        """
        Returns the precise BPE token count for a string.
        """
        if not text:
            return 0
        return estimate_tokens(text, self.model_name)

    def calculate_pareto_score(self, accuracy: float, token_count: int) -> float:
        """
        Computes fitness: Score = (Accuracy * Alpha) - (Tokens * Beta)
        """
        return (accuracy * self.alpha) - (token_count * self.beta)

class LatencyTimer:
    """
    Context manager for measuring wall-clock execution time.
    """
    def __init__(self):
        self.start = 0.0
        self.end = 0.0
        self.duration_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.duration_ms = (self.end - self.start) * 1000.0