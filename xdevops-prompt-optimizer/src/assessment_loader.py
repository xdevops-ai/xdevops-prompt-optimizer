import json
import random
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Dataset:
    train_set: List[Dict]
    test_set: List[Dict]

class AssessmentLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_data = []

    def load(self) -> None:
        """
        Loads the JSON file and validates the basic schema.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Assessment file not found at: {self.filepath}")

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Assessment file at {self.filepath} is not valid JSON.")

        self._validate_structure()

    def _validate_structure(self) -> None:
        """
        Ensures every entry has the required keys.
        """
        if not isinstance(self.raw_data, list):
            raise ValueError("Assessment file must be a JSON list of objects.")

        for index, item in enumerate(self.raw_data):
            if "conversation" not in item:
                raise ValueError(f"Item at index {index} missing required key: 'conversation'")
            if "expected_json" not in item:
                raise ValueError(f"Item at index {index} missing required key: 'expected_json'")
            
            # Ensure expected_json is a string or dict
            if isinstance(item["expected_json"], str):
                try:
                    json.loads(item["expected_json"])
                except json.JSONDecodeError:
                    raise ValueError(f"Item at index {index} has invalid JSON string in 'expected_json'")

    def split_data(self, train_ratio: float = 0.8, seed: int = 42) -> Dataset:
        """
        Shuffles and splits the data into Training and Test sets.
        Uses a fixed seed for reproducibility.
        """
        if not self.raw_data:
            raise ValueError("Data not loaded. Call load() first.")

        data_copy = self.raw_data.copy()
        
        # Deterministic shuffle
        random.seed(seed)
        random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        
        train_set = data_copy[:split_index]
        test_set = data_copy[split_index:]

        # Safety fallback for very small datasets
        if len(data_copy) > 1 and (not train_set or not test_set):
            mid = len(data_copy) // 2
            train_set = data_copy[:mid]
            test_set = data_copy[mid:]

        return Dataset(train_set=train_set, test_set=test_set)