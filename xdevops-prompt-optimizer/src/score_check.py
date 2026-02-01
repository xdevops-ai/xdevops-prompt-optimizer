import os
from config import Config
from metrics import MetricsEngine, LatencyTimer

def run_scorer_test():
    print("--- Starting Scorer & Config Check ---")

    # 1. Config Validation
    print("\n[1/3] Validating Configuration...")
    try:
        # Check if API key exists (even if dummy for this test)
        if not os.getenv("OPENAI_API_KEY"):
            print("WARN: OPENAI_API_KEY not found in env. Set it for later steps.")
        
        Config.validate()
        print(f"PASS: Config loaded. Alpha={Config.ALPHA_ACCURACY}, Beta={Config.BETA_TOKEN_PENALTY}")
    except ValueError as e:
        print(f"FAIL: Config validation error: {e}")
    
    # 2. Tokenizer Check
    print("\n[2/3] Testing Tokenizer (tiktoken)...")
    engine = MetricsEngine()
    test_str = "The quick brown fox jumps over the lazy dog."
    count = engine.count_tokens(test_str)
    
    # "The quick brown fox jumps over the lazy dog." is usually 9 or 10 tokens depending on model
    print(f"Input: '{test_str}'")
    print(f"Token Count: {count}")
    
    if count > 0:
        print("PASS: Tokenizer is functioning.")
    else:
        print("FAIL: Tokenizer returned 0.")

    # 3. Pareto Formula Check
    print("\n[3/3] Testing Pareto Scoring Formula...")
    # Scenario A: Perfect Accuracy, Heavy Prompt
    acc_a = 1.0
    tokens_a = 1000
    score_a = engine.calculate_pareto_score(acc_a, tokens_a)
    
    # Scenario B: Perfect Accuracy, Lean Prompt (Optimized)
    acc_b = 1.0
    tokens_b = 800
    score_b = engine.calculate_pareto_score(acc_b, tokens_b)

    print(f"Scenario A (1000 tokens): Score {score_a}")
    print(f"Scenario B (800 tokens):  Score {score_b}")

    if score_b > score_a:
        print("PASS: Scoring correctly favors efficiency (Lower tokens = Higher score).")
    else:
        print("FAIL: Scoring logic is broken (Efficiency was not rewarded).")

    # 4. Latency Timer Check
    print("\n[4/4] Testing Latency Timer...")
    with LatencyTimer() as t:
        # Simulate work
        dummy = [x**2 for x in range(10000)]
    
    print(f"Operation took {t.duration_ms:.2f}ms")
    if t.duration_ms > 0:
        print("PASS: Timer recorded duration.")
    else:
        print("FAIL: Timer reported 0ms.")

if __name__ == "__main__":
    run_scorer_test()