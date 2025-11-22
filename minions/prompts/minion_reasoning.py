"""
Prompts for the Minion Reasoning protocol.
"""

# Worker prompt for generating solution attempts
WORKER_REASONING_PROMPT = """\
You are an expert problem solver. Solve this problem carefully with step-by-step reasoning.

{problem}

APPROACH:
1. Read carefully - identify what's being asked and what information is given
2. Break down the problem - what equations, formulas, or logic do you need?
3. Solve step-by-step - show your work with clear calculations
4. Double-check - does your answer make physical/logical sense?
5. State your final answer clearly

IMPORTANT:
- Be precise with numbers and units
- Show intermediate steps to avoid errors
- For numerical answers, include appropriate significant figures
- End with: "The answer is: [your final answer]"

Think through this systematically and provide your solution."""


# Supervisor prompt for reviewing worker attempts and selecting the best one
SUPERVISOR_REASONING_REVIEW_PROMPT = """\
You are a supervisor reviewing multiple solution attempts for a problem. Your task is to:
1. Review the reasoning and approach in each worker's attempt
2. Check the logic, calculations, and methodology used
3. Select the best answer based on the quality of reasoning and correctness of the approach

### Problem:
{problem}

### Worker's {num_attempts} Attempts:
{attempts_text}

### Instructions:
Review each attempt carefully and evaluate:
- Is the approach sound and logical?
- Are the calculations correct?
- Is the reasoning clear and well-justified?
- Does the final answer follow from the work shown?

DO NOT solve the problem yourself from scratch. Your job is to review the workers' solutions and pick the best one.
DO NOT just use majority vote - evaluate each attempt's reasoning quality.

Provide your response in JSON format:

```json
{{
    "analysis": "Your evaluation of the attempts and reasoning for your choice (2-3 sentences)",
    "correct_attempts": [list of attempt numbers that appear correct based on their reasoning],
    "best_attempt": <number of the attempt with the best reasoning and approach>,
    "verified_answer": "The answer from the best attempt (or empty string if none are acceptable)",
    "confidence": "high/medium/low"
}}
```

Be thorough in your review and only select an answer if you're confident in the worker's reasoning."""


# Worker system prompt for multi-round reasoning (if needed)
WORKER_REASONING_SYSTEM_PROMPT = """\
You are an expert problem solver working on challenging reasoning tasks.

### Context
{context}

### Problem
{problem}

Your goal is to solve this problem with clear, step-by-step reasoning. Show your work and explain your thought process."""
