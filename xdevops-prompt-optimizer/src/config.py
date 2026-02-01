import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # --- Provider Selection ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    # --- Authentication ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # --- Model Selection ---
    # Fast model for Discriminator/validation (if needed)
    MODEL_FAST = os.getenv("MODEL_FAST", "gpt-4o-mini")
    # Smart model for Phase 1 (Architect) & Phase 2 (Surgeon)
    MODEL_SMART = os.getenv("MODEL_SMART", "gpt-4o")
    
    # Minimum token reduction required to accept a change
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.100"))
    
    # --- Optimization Hyperparameters ---
    # How many iterations to wait without improvement before stopping
    PATIENCE = int(os.getenv("PATIENCE", "3"))
    ARCHITECT_PATIENCE = int(os.getenv("ARCHITECT_PATIENCE", "6"))
    
    # Pareto Weights (Fitness Function)
    # Alpha: Weight for Accuracy (High = Accuracy is paramount)
    ALPHA_ACCURACY = float(os.getenv("ALPHA_ACCURACY", "100.0"))
    # Beta: Penalty per token (0.01 = 100 tokens cost 1.0 score point)
    BETA_TOKEN_PENALTY = float(os.getenv("BETA_TOKEN_PENALTY", "0.01"))
    
    # --- LLM Temperatures ---
    TEMPERATURE_VALIDATOR = float(os.getenv("TEMPERATURE_VALIDATOR", "0.0"))
    TEMPERATURE_ARCHITECT = float(os.getenv("TEMPERATURE_ARCHITECT", "0.1"))
    TEMPERATURE_EFFICIENCY = float(os.getenv("TEMPERATURE_EFFICIENCY", "0.5"))

    # --- Paths ---
    ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", os.getenv("ASSETS_DIR", "assets"))
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimized")

    SYSTEM_PROMPT_PATH = ASSETS_DIR + "/system_prompt.json"
    ASSESSMENT_PATH = ASSETS_DIR + "/assessment.json"
    META_PROMPT_PATH = ASSETS_DIR + "/meta_prompt.txt"
    META_PROMPT_EFFICIENCY_PATH = ASSETS_DIR + "/meta_prompt_efficiency.txt"
    
    @staticmethod
    def validate():
        """
        Ensures critical configuration is present.
        """
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        for name, value in [
            ("TEMPERATURE_VALIDATOR", Config.TEMPERATURE_VALIDATOR),
            ("TEMPERATURE_ARCHITECT", Config.TEMPERATURE_ARCHITECT),
            ("TEMPERATURE_EFFICIENCY", Config.TEMPERATURE_EFFICIENCY)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")