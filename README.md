# xDevOps Prompt Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Challenge](https://img.shields.io/badge/Challenge-Active-brightgreen.svg)]()

> **A deterministic, self-healing, and Pareto-optimized engineering pipeline for LLM System Prompts.**

---

## 🚀 The xDevOps Architect Challenge

**We are looking for a Lead Architect. Is it you?**

We have open-sourced our internal prompt optimization engine. It is powerful, but it has architectural flaws (vendor lock-in, code duplication). We are betting that the open-source community can engineer it better than we can.

### 📅 Contest Timeline
- **Start Date:** February 1st, 2026
- **Deadline:** March 1st, 2026 (Midnight CET)

### 🏆 The Prize
The developer with the highest **Pareto Score** (Accuracy vs. Efficiency) and cleanest architecture wins:
- ✈️ **Round-trip Flight** (from Europe) + **Hotel** in Bucharest, Romania 🇷🇴.
- 🤝 **VIP Weekend** with the xDevOps team to discuss the future of AI Governance.
- 💰 **$200 Signing Bonus** + **"The Term"** (A 1-month paid trial contract with full-time potential).

### 🎯 The Mission
Your goal is to refactor this repository to achieve **Model Agnosticism** and **Maximum Efficiency**.

1. **Unlock the Factory:** The `llm_engine` supports Gemini and DeepSeek, but `src/llm_client.py` is currently vendor-locked to OpenAI. **Fix it.**
2. **Optimize Efficiency:** Improve the "Surgeon" agent in `src/optimizer.py` to reduce token usage without dropping accuracy below 100%.
3. **Enforce DRY:** `src/metrics.py` duplicates logic found in `llm_engine/capabilities.py`. **Refactor it.**

### 📥 How to Enter
1. **Fork** this repository to your own GitHub account.
2. **Clone** your fork and create a new branch (e.g., `feature/architect-challenge`).
3. **Refactor & Optimize** the code according to the mission above.
4. **Submit a Pull Request (PR)** to this main repository with the tag `[CONTEST]` in the title before **March 1st, 2026**.

---

## 🛠️ How It Works

This is not a creative writing tool. It is an engineering harness designed to force JSON-outputting system prompts to converge on **100% Reliability**.

It operates in a **Dual-Phase Evolutionary Loop**:

1. **Phase 1 (The Architect):**
   - Focus: **Accuracy**.
   - Iteratively repairs the prompt based on validation failures until it hits 100% on the training set.
2. **Phase 2 (The Surgeon):**
   - Focus: **Efficiency**.
   - Uses a "Hill Climbing" algorithm to prune tokens.
   - **Rollback Mechanism:** If accuracy drops below 100%, the change is immediately reverted.

### The Pareto Score
We judge performance based on this formula:

$$\text{Score} = (\text{Accuracy} \times \alpha) - (\text{Token Count} \times \beta)$$

---

## 📂 Repository Structure

```text
xdevops-prompt-optimizer
├── assets/                  # The "Soul" of the system (Data & Prompts)
│   ├── assessment.json      # Ground Truth test cases (Train/Test split)
│   ├── system_prompt.json   # The Candidate Prompt to be optimized
│   └── meta_prompt*.txt     # Instructions for the AI Agents
├── llm_engine/              # The Core Library (Providers & Factory)
│   ├── factory.py           # Logic to switch between OpenAI/Gemini/DeepSeek
│   └── providers/           # API Wrappers
├── src/                     # The Application Logic (Orchestrator)
│   ├── main.py              # Entry point
│   ├── optimizer.py         # The AI Agents (Architect & Efficiency Expert)
│   └── validator.py         # Deterministic JSON Validation Engine
└── requirements.txt         # Dependencies
```

---

## ⚡ Quick Start

### Clone & Install

```bash
# Clone your fork!
git clone https://github.com/YOUR_USERNAME/xdevops-prompt-optimizer.git
cd xdevops-prompt-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file in the root:

```ini
# Providers
LLM_PROVIDER=openai  # Options: openai, gemini, deepseek

# Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Tuning
ALPHA_ACCURACY=100.0
BETA_TOKEN_PENALTY=0.01
```

### Run the Optimizer

```bash
cd src
python main.py
```

---

## 📜 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

> **Note to Contestants:** We value clean architecture as much as raw performance. A fast hacky solution will lose to a clean, maintainable, and robust solution.

