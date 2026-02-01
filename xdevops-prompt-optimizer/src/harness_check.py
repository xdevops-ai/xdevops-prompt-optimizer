import json
import os
from validator import Validator
from assessment_loader import AssessmentLoader

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets")
ASSESSMENT_FILE = os.path.join(ASSETS_DIR, "assessment.json")
SYSTEM_PROMPT_FILE = os.path.join(ASSETS_DIR, "system_prompt.json")

def run_integrity_check():
    print("--- Starting Harness Integrity Check ---")

    # 1. System Prompt Check
    print(f"\n[1/3] Checking System Prompt ({SYSTEM_PROMPT_FILE})...")
    if not os.path.exists(SYSTEM_PROMPT_FILE):
        print("FAIL: System prompt file missing.")
        return
    
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
            print("PASS: System prompt is valid JSON.")
            # Basic schema check
            if "system_prompt" in prompt_data:
                print("PASS: Found 'system_prompt' root key.")
            else:
                print("WARN: Root key 'system_prompt' missing.")
    except json.JSONDecodeError as e:
        print(f"FAIL: System prompt JSON error: {e}")
        return

    # 2. Assessment Data Check
    print(f"\n[2/3] Checking Assessment Data ({ASSESSMENT_FILE})...")
    loader = AssessmentLoader(ASSESSMENT_FILE)
    try:
        loader.load()
        print(f"PASS: Loaded {len(loader.raw_data)} test cases.")
    except Exception as e:
        print(f"FAIL: Assessment load error: {e}")
        return

    # 3. Validator Logic Check (Self-Test)
    print("\n[3/3] Running Validator Self-Test on Data...")
    validator = Validator()
    dataset = loader.split_data(train_ratio=0.8)
    print(f"INFO: Split data into {len(dataset.train_set)} Train / {len(dataset.test_set)} Test")

    valid_count = 0
    for i, item in enumerate(dataset.train_set):
        expected_str = item["expected_json"]
        
        # In a real run, 'actual' comes from the LLM. 
        # Here, we test if the EXPECTED string is actually parseable by our tools.
        parsed_expected = validator.parse_json(expected_str)
        
        if parsed_expected is None:
            print(f"FAIL: Test case #{i} 'expected_json' is not valid JSON.")
        else:
            # Trivial check: validate expected against itself to ensure logic holds
            result = validator.validate(parsed_expected, parsed_expected)
            if result.passed:
                valid_count += 1
            else:
                print(f"FAIL: Validator failed to match identical objects at index {i}. Error: {result.error_message}")

    print(f"\nSummary: {valid_count}/{len(dataset.train_set)} training items verified internal integrity.")
    
    if valid_count == len(dataset.train_set):
        print("\nSUCCESS: Infrastructure is robust. Ready for Step 2 (Metrics).")
    else:
        print("\nWARNING: Some assessment items are malformed.")

if __name__ == "__main__":
    run_integrity_check()