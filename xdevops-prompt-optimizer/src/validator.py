import json
import math
import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass

logger = logging.getLogger("GoProValidator")

@dataclass
class ValidationResult:
    passed: bool
    error_message: Optional[str] = None

class Validator:
    def __init__(self, unordered_paths: Optional[Set[str]] = None):
        """
        Initialize with a set of paths that should be treated as unordered sets.
        :param unordered_paths: A set of strings like {"options.filters.tags", "options.filters.user_id"}
        """
        self.unordered_paths = unordered_paths or set()

    @staticmethod
    def parse_json(response_text: str) -> Optional[Dict[str, Any]]:
        """
        Clean and parse a JSON string, handling Markdown code fences.
        """
        if not response_text:
            return None
            
        cleaned_text = response_text.strip()
        
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        try:
            return json.loads(cleaned_text.strip())
        except json.JSONDecodeError:
            return None

    def validate(self, actual: Any, expected: Any, path: str = "") -> ValidationResult:
        """
        Recursively compares actual vs expected values with domain-specific logic.
        """
        # 1. Type Mismatch Check
        if type(actual) != type(expected):
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                pass
            else:
                return ValidationResult(False, f"Type mismatch at '{path}': expected {type(expected).__name__}, got {type(actual).__name__}")

        # 2. Dictionary Comparison
        if isinstance(expected, dict):
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())
            
            if expected_keys != actual_keys:
                missing = expected_keys - actual_keys
                extra = actual_keys - expected_keys
                return ValidationResult(False, f"Key mismatch at '{path}'. Missing: {list(missing)}, Extra: {list(extra)}")
            
            for key in expected:
                current_path = f"{path}.{key}" if path else key
                result = self.validate(actual[key], expected[key], current_path)
                if not result.passed:
                    return result

        # 3. List Comparison (The Dynamic Logic)
        elif isinstance(expected, list):
            if len(expected) != len(actual):
                return ValidationResult(False, f"List length mismatch at '{path}': expected {len(expected)}, got {len(actual)}")

            # Check if this specific path is in our injected configuration
            is_unordered = path in self.unordered_paths

            if is_unordered:
                # Set Logic: Order does not matter
                try:
                    # Convert to strings to ensure hashability for set comparison
                    expected_set = sorted([str(x) for x in expected])
                    actual_set = sorted([str(x) for x in actual])
                    if expected_set != actual_set:
                        return ValidationResult(False, f"Set content mismatch at '{path}': {expected_set} != {actual_set}")
                except Exception as e:
                     return ValidationResult(False, f"Unordered comparison failed at '{path}': {e}")
            else:
                # Sequence Logic: Order matters
                for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                    result = self.validate(act_item, exp_item, f"{path}[{i}]")
                    if not result.passed:
                        return result

        # 4. Float Comparison
        elif isinstance(expected, float):
            if not math.isclose(actual, expected, rel_tol=1e-3):
                return ValidationResult(False, f"Float mismatch at '{path}': expected {expected}, got {actual}")

        # 5. Primitive Value Comparison
        else:
            if actual != expected:
                return ValidationResult(False, f"Value mismatch at '{path}': expected '{expected}', got '{actual}'")

        return ValidationResult(True)