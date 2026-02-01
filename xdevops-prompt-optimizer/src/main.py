import asyncio
import json
import os
import logging
import sys

# Ensure llm_engine is importable globally
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from typing import List, Dict, Tuple
from datetime import datetime

# Component Imports
from config import Config
from assessment_loader import AssessmentLoader
from validator import Validator, ValidationResult
from metrics import MetricsEngine
from llm_client import AsyncLLMClient
from optimizer import Architect, EfficiencyExpert

# Setup Logging
# Ensure configuration is valid and output directories exist
Config.validate()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.OUTPUT_DIR, "optimization.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GoProOrchestrator")

class Orchestrator:
    def __init__(self):
        self.config = Config()
        self.loader = AssessmentLoader(Config.ASSESSMENT_PATH)
        self.validator = Validator()
        self.metrics = MetricsEngine()
        self.llm_client = AsyncLLMClient()
        
        # The AI Agents
        self.architect = Architect(self.llm_client)
        self.expert = EfficiencyExpert(self.llm_client)
        
        # State
        self.best_prompt = ""
        self.dataset = None

    async def initialize(self):
        """Loads data and initial system prompt."""
        logger.info("--- Initializing GoPro Optimizer ---")

        # 1. Load Validation Rules (Dynamic Configuration)
        validation_config_path = os.path.join(Config.ASSETS_DIR, "validation_rules.json")
        unordered_paths = set()
        
        if os.path.exists(validation_config_path):
            try:
                with open(validation_config_path, "r") as f:
                    rules = json.load(f)
                    unordered_paths = set(rules.get("unordered_paths", []))
                logger.info(f"Loaded {len(unordered_paths)} unordered validation paths.")
            except Exception as e:
                logger.warning(f"Failed to load validation rules: {e}")
        
        # INJECT the config into the Validator
        self.validator = Validator(unordered_paths=unordered_paths)
        
        # 2. Load and Split Data
        self.loader.load()
        self.dataset = self.loader.split_data(train_ratio=0.8)
        logger.info(f"Data Loaded: {len(self.dataset.train_set)} Training / {len(self.dataset.test_set)} Holdout")
        
        # 3. Load Initial Prompt
        if not os.path.exists(Config.SYSTEM_PROMPT_PATH):
            raise FileNotFoundError(f"System prompt not found at {Config.SYSTEM_PROMPT_PATH}")
        
        with open(Config.SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            self.best_prompt = f.read()
            
        logger.info(f"Initial Prompt Size: {self.metrics.count_tokens(self.best_prompt)} tokens")

    def _save_result(self, filename="system_prompt_optimized.json"):
        """Helper to save the current best prompt to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            final_filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(Config.OUTPUT_DIR, final_filename)
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(self.best_prompt)
            current_tokens = self.metrics.count_tokens(self.best_prompt)
            logger.info(f"Saved best prompt to '{filepath}' ({current_tokens} tokens).")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def evaluate_batch(self, prompt: str, data: List[Dict]) -> Tuple[float, List[Dict], float]:
        """
        Runs the Validator against a specific dataset (Train or Test).
        Returns: (Accuracy 0.0-1.0, List of Failure Logs, Avg Output Tokens)
        """
        return asyncio.run(self._evaluate_batch_async(prompt, data)) 

    async def _evaluate_batch_async(self, prompt: str, data: List[Dict]) -> Tuple[float, List[Dict], float]:
        failures = []
        passed_count = 0
        total_valid_tokens = 0
        
        # We process in small batches to avoid Rate Limits
        BATCH_SIZE = 5
        
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            tasks = []
            
            # 1. Dispatch Batch Requests
            for item in batch:
                tasks.append(self.llm_client.generate_response(
                    system_prompt=prompt,
                    user_message=item['conversation'][-1]['content'],
                    model=Config.MODEL_FAST,
                    temperature=Config.TEMPERATURE_VALIDATOR
                ))
            
            # 2. Await Results
            results = await asyncio.gather(*tasks)
            
            # 3. Validate & Log
            for idx, (item, actual_response) in enumerate(zip(batch, results)):
                # Parse expected JSON safely
                expected = item['expected_json']
                if isinstance(expected, str):
                    try:
                        expected = json.loads(expected)
                    except:
                        logger.error(f"DATA ERROR: Could not parse expected JSON for item {i+idx}")

                # Run Validator
                validation = self.validator.validate(actual_response, expected)
                
                if validation.passed:
                    passed_count += 1
                    total_valid_tokens += self.metrics.count_tokens(json.dumps(actual_response))
                else:
                    logger.warning(f"\n[FAILURE DETECTED] Test Case #{i + idx + 1}")
                    logger.warning(f"INPUT:    {item['conversation'][-1]['content']}")
                    logger.warning(f"EXPECTED: {json.dumps(expected)}")
                    logger.warning(f"ACTUAL:   {json.dumps(actual_response)}")
                    logger.warning(f"REASON:   {validation.error_message}")
                    logger.warning("-" * 50)

                    failures.append({
                        "input": item['conversation'][-1]['content'],
                        "expected": expected,
                        "actual": actual_response,
                        "error_message": validation.error_message
                    })

        # Calculate average output tokens for valid responses
        avg_output_tokens = 0.0
        if passed_count > 0:
            avg_output_tokens = total_valid_tokens / passed_count

        accuracy = passed_count / len(data) if data else 0.0
        return accuracy, failures, avg_output_tokens

    async def run_pipeline(self):
        try:
            await self.initialize()
            
            # --- PHASE 1: THE ARCHITECT (ACCURACY) ---
            logger.info("\n=== PHASE 1: ARCHITECT (Repair Loop) ===")
            iteration = 0
            
            # Track metrics to avoid re-evaluation in Phase 2
            current_accuracy = 0.0
            avg_output_tokens = 0.0
            
            while True:
                iteration += 1
                logger.info(f"Phase 1 - Iteration {iteration}")
                
                # Test against Training Set
                accuracy, failures, avg_out = await self._evaluate_batch_async(self.best_prompt, self.dataset.train_set)
                logger.info(f"Accuracy: {accuracy:.1%} ({len(failures)} failures)")
                
                current_accuracy = accuracy
                avg_output_tokens = avg_out

                if accuracy == 1.0:
                    logger.info("Phase 1 Complete: 100% Accuracy achieved.")
                    break
                
                if iteration > Config.ARCHITECT_PATIENCE:
                    logger.warning("Max iterations reached in Phase 1. Stopping.")
                    break
                    
                # Invoke Architect
                logger.info("Requesting repairs from Architect...")
                self.best_prompt = await self.architect.repair_prompt(self.best_prompt, failures)

            # --- PHASE 2: THE EFFICIENCY EXPERT (COMPRESSION) ---
            logger.info("\n=== PHASE 2: EFFICIENCY EXPERT (Compression Loop) ===")
            
            # Baseline metrics
            input_tokens = self.metrics.count_tokens(self.best_prompt)
            
            # Total "Cost" Metric: Input + (Output * 2) to weight generation latency higher? 
            # For now, let's just sum them as "Total Transaction Tokens"
            current_total_tokens = input_tokens + avg_output_tokens
            
            current_score = self.metrics.calculate_pareto_score(current_accuracy, current_total_tokens)
            patience_counter = 0
            
            while patience_counter < Config.PATIENCE:
                logger.info(f"Phase 2 - Total Tokens: {current_total_tokens:.1f} (In: {input_tokens} + Out: {avg_output_tokens:.1f}) | Score: {current_score:.4f}")
                
                # Invoke Efficiency Expert
                candidate_prompt = await self.expert.optimize_prompt(self.best_prompt)
                candidate_input_tokens = self.metrics.count_tokens(candidate_prompt)
                
                # Validate Candidate
                accuracy, failures, cand_avg_out = await self._evaluate_batch_async(candidate_prompt, self.dataset.train_set)
                
                candidate_total_tokens = candidate_input_tokens + cand_avg_out
                candidate_score = self.metrics.calculate_pareto_score(accuracy, candidate_total_tokens)

                score_diff = candidate_score - current_score

                # Enforce 100% Accuracy AND Pareto Improvement
                if accuracy == 1.0 and score_diff > Config.SCORE_THRESHOLD:
                    logger.info(f"Pareto Improvement! Score {current_score:.4f} -> {candidate_score:.4f} (+{score_diff:.4f})")
                    self.best_prompt = candidate_prompt
                    current_total_tokens = candidate_total_tokens
                    input_tokens = candidate_input_tokens
                    avg_output_tokens = cand_avg_out
                    current_accuracy = accuracy
                    current_score = candidate_score
                    patience_counter = 0
                    self._save_result()
                else:
                    if accuracy < 1.0:
                        logger.warning(f"Candidate rejected. Accuracy dropped to {accuracy:.1%} (Must be 100%). Reverting.")
                    else:
                        logger.warning(f"Candidate rejected. Score diff {score_diff:.4f} <= Threshold {Config.SCORE_THRESHOLD} (Score: {candidate_score:.4f}). Reverting.")
                    patience_counter += 1

            # --- PHASE 3: THE GATEKEEPER (HOLDOUT TEST) ---
            logger.info("\n=== PHASE 3: GATEKEEPER (Final Validation) ===")
            
            test_accuracy, test_failures, _ = await self._evaluate_batch_async(self.best_prompt, self.dataset.test_set)
            
            logger.info(f"Final Test Set Accuracy: {test_accuracy:.1%}")
            
            if test_accuracy == 1.0:
                logger.info("RESULT: PASS. System is robust.")
            elif test_accuracy >= 0.95:
                logger.info("RESULT: PASS (With Warnings). Minor generalization errors.")
            else:
                logger.warning("RESULT: FAIL. High overfitting detected.")
                
            # Save Result
            self._save_result()
        except KeyboardInterrupt:
            logger.warning("Orchestrator interrupted by user.")
            logger.info("Saving current best prompt before exit...")
            self._save_result()
        except Exception as e:
            logger.error(f"Orchestrator encountered a critical error: {e}")

if __name__ == "__main__":
    orchestrator = Orchestrator()
    asyncio.run(orchestrator.run_pipeline())