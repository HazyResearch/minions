import os
import json
import openai
import requests
from typing import List, Dict, Any
import numpy as np
import time
from ollama import chat
import argparse


# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


class Evaluator:
    def __init__(self, gru_model="gpt-4o", minion_model="llama3.2:1b"):
        """Initialize the evaluator with teacher and student models."""
        self.gru_model = gru_model
        self.minion_model = minion_model
        self.gru_client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Set up OpenAI client
        openai.api_key = OPENAI_API_KEY

    def query_gpt4o(self, prompt: str) -> str:
        """Query the teacher model (GPT-4o)."""

        try:
            response = self.gru_client.chat.completions.create(
                model=self.gru_model,  # Use the specified teacher model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying {self.gru_model}: {e}")
            return ""

    def query_ollama(self, prompt: str, model_name: str = None) -> str:
        """Query the student model (Ollama)."""
        model_to_use = model_name or self.minion_model
        response = chat(
            model=model_to_use,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.message.content

    def analyze_task_for_skills(
        self, task_description: str, document: str
    ) -> List[Dict[str, Any]]:
        """Have GPT-4o analyze the task to determine required skills."""
        prompt = f"""
        You are an AI teacher tasked with evaluating the abilities of a small local model. You plan on decomposing the task into small sub-tasks that these small local models can handle.
        
        TASK DESCRIPTION:
        {task_description}
        
        DOCUMENT SAMPLE:
        {document}  # Using first 2000 chars as sample
        
        Based on this task and document, decompose the task into smaller sub-tasks, and identify the key skills required to successfully complete each sub-task.
        For each skill:
        1. Name the skill
        2. Describe what the skill entails
        3. Explain why it's needed for this task
        4. Suggest how to test this skill with 2-3 concrete examples
        
        Format your response as a JSON array of skills with these fields:
        - skill_name: The name of the skill
        - description: What the skill entails
        - relevance: Why it's needed for this task
        - test_approach: How to test this skill
        
        Return valid JSON only.
        """

        response = self.query_gpt4o(prompt)

        # Safely parse the JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response

            skills = json.loads(json_str)
            return skills
        except Exception as e:
            print(f"Error parsing skills JSON: {e}")
            print(f"Raw response: {response}")
            # Return a default skill set if parsing fails
            return [
                {
                    "skill_name": "Information Retrieval",
                    "description": "Ability to find and extract relevant information from text",
                    "relevance": "Needed to locate specific data points in the document",
                    "test_approach": "Ask the model to find specific information in the text",
                },
                {
                    "skill_name": "Numerical Computation",
                    "description": "Ability to perform calculations based on extracted information",
                    "relevance": "Needed to compute changes or perform required calculations",
                    "test_approach": "Provide numerical data and ask for specific calculations",
                },
            ]

    def generate_skill_test(
        self, skill: Dict[str, Any], document: str
    ) -> List[Dict[str, Any]]:
        """Have GPT-4o generate specific test problems for a given skill."""
        prompt = f"""
        Based on the following skill and document, generate 3 specific test problems with ground truth answers.
        
        SKILL:
        {json.dumps(skill, indent=2)}
        
        DOCUMENT:
        {document[:3000]}  # Using first 3000 chars
        
        For each test problem:
        1. Create a clear, specific question that tests the skill
        2. Provide the ground truth answer
        3. Include evaluation criteria to determine if a response is correct
        
        Create mock data if needed that resembles the document style but with simplified/fictional values.
        
        Format your response as a JSON array with these fields for each problem:
        - question: The test question
        - context: The mock document or context to use (if needed)
        - ground_truth: The correct answer
        - evaluation_criteria: Specific criteria to determine correctness
        
        Return valid JSON only.
        """

        response = self.query_gpt4o(prompt)

        # Parse the JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response

            test_problems = json.loads(json_str)
            return test_problems
        except Exception as e:
            print(f"Error parsing test problems JSON: {e}")
            print(f"Raw response: {response}")
            # Return a default test problem if parsing fails
            return [
                {
                    "question": f"Based on the document, what is an example of {skill['skill_name']}?",
                    "context": "Mock data would be here.",
                    "ground_truth": "Sample answer based on the skill.",
                    "evaluation_criteria": "Response should demonstrate understanding of the concept.",
                }
            ]

    def evaluate_student_response(
        self, skill: Dict[str, Any], test_problem: Dict[str, Any], student_response: str
    ) -> Dict[str, Any]:
        """Have GPT-4o evaluate the student model's response to a test problem."""
        prompt = f"""
        Evaluate the student model's response to this test problem.
        
        SKILL BEING TESTED:
        {json.dumps(skill, indent=2)}
        
        TEST PROBLEM:
        {json.dumps(test_problem, indent=2)}
        
        STUDENT'S RESPONSE:
        {student_response}
        
        Evaluate the response based on the evaluation criteria. Give a score from 0 to 5 where:
        0 = Completely incorrect/irrelevant
        1 = Shows minimal understanding but major errors
        2 = Partially correct but significant gaps
        3 = Mostly correct with minor issues
        4 = Correct with very minor imperfections
        5 = Perfect/optimal response
        
        Format your response as a JSON object with these fields:
        - score: Numerical score (0-5)
        - justification: Brief explanation of the score
        - feedback: Specific feedback about strengths and weaknesses
        
        Return valid JSON only.
        """

        response = self.query_gpt4o(prompt)

        # Parse the JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response

            evaluation = json.loads(json_str)
            return evaluation
        except Exception as e:
            print(f"Error parsing evaluation JSON: {e}")
            print(f"Raw response: {response}")
            # Return a default evaluation if parsing fails
            return {
                "score": 2.5,  # Neutral score
                "justification": "Unable to properly evaluate response.",
                "feedback": "The evaluation system encountered an error analyzing this response.",
            }

    def generate_skill_report(
        self, skill: Dict[str, Any], evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a summary report for a specific skill based on all test evaluations."""
        avg_score = np.mean([e["score"] for e in evaluations])

        skill_level = (
            "Excellent"
            if avg_score >= 4.5
            else (
                "Good"
                if avg_score >= 3.5
                else (
                    "Adequate"
                    if avg_score >= 2.5
                    else "Needs Improvement" if avg_score >= 1.5 else "Poor"
                )
            )
        )

        return {
            "skill_name": skill["skill_name"],
            "average_score": round(float(avg_score), 2),
            "proficiency_level": skill_level,
            "evaluations": evaluations,
            "strengths": [
                (
                    e["feedback"]
                    .split("Strengths:")[-1]
                    .split("Weaknesses:")[0]
                    .strip()
                    if "Strengths:" in e["feedback"] and "Weaknesses:" in e["feedback"]
                    else ""
                )
                for e in evaluations
                if e["score"] >= 3
            ],
            "weaknesses": [
                (
                    e["feedback"].split("Weaknesses:")[-1].strip()
                    if "Weaknesses:" in e["feedback"]
                    else ""
                )
                for e in evaluations
                if e["score"] < 4
            ],
        }

    def run_evaluation(
        self,
        task_description: str,
        document: str,
    ) -> Dict[str, Any]:
        """Run the full evaluation process and generate a comprehensive report."""
        print("Analyzing task for required skills...")
        skills = self.analyze_task_for_skills(task_description, document)

        # Remove the breakpoint and fix the skills structure handling
        all_skill_reports = []

        # Handle both possible structures of the skills response
        if isinstance(skills, list):
            skills_list = skills
        elif isinstance(skills, dict) and "skills" in skills:
            skills_list = skills["skills"]
        else:
            skills_list = [skills]  # Fallback if structure is unexpected

        for skill in skills_list:
            print(f"Evaluating skill: {skill['skill_name']}...")

            # Generate test problems for this skill
            test_problems_response = self.generate_skill_test(skill, document)

            # Handle both possible structures of the test_problems response
            if isinstance(test_problems_response, list):
                test_problems = test_problems_response
            elif (
                isinstance(test_problems_response, dict)
                and "test_problems" in test_problems_response
            ):
                test_problems = test_problems_response["test_problems"]
            else:
                test_problems = [test_problems_response]  # Fallback

            skill_evaluations = []

            for i, problem in enumerate(test_problems):
                print(f"  Running test problem {i+1}/{len(test_problems)}...")

                # Create prompt for the student model
                student_prompt = f"""
                {problem['context'] if 'context' in problem else document[:2000]}
                
                Question: {problem['question']}
                
                Please answer the question based on the provided information.
                """

                # Get student model's response
                student_response = self.query_ollama(student_prompt)

                # Evaluate the response
                evaluation = self.evaluate_student_response(
                    skill, problem, student_response
                )

                # Store the evaluation along with the problem and response
                full_evaluation = {
                    **evaluation,
                    "problem": problem["question"],
                    "student_response": student_response,
                }
                skill_evaluations.append(full_evaluation)

                # Add a small delay to avoid rate limits
                time.sleep(1)

            # Generate skill report
            skill_report = self.generate_skill_report(skill, skill_evaluations)
            all_skill_reports.append(skill_report)

        # Generate final report
        overall_score = np.mean(
            [report["average_score"] for report in all_skill_reports]
        )

        final_report = {
            "task": task_description,
            "overall_score": round(float(overall_score), 2),
            "overall_assessment": (
                "Excellent"
                if overall_score >= 4.5
                else (
                    "Good"
                    if overall_score >= 3.5
                    else (
                        "Adequate"
                        if overall_score >= 2.5
                        else "Needs Improvement" if overall_score >= 1.5 else "Poor"
                    )
                )
            ),
            "skill_reports": all_skill_reports,
        }

        return final_report


def read_document(file_path):
    """Read document from file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading document file: {e}")
        return None


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate a student model on a specific task."
    )
    parser.add_argument(
        "--context_path", type=str, required=True, help="Path to the context file"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Description of the task to evaluate"
    )
    parser.add_argument(
        "--gru_model",
        type=str,
        default="gpt-4o-mini",
        help="Teacher model to use for evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--minion_model",
        type=str,
        default="llama3.2:1b",
        help="Student model to evaluate (default: llama3.2:1b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="skill_evaluation_report.json",
        help="Output file path for the evaluation report (default: skill_evaluation_report.json)",
    )

    args = parser.parse_args()

    # Read document from file
    context = read_document(args.context_path)
    if not context:
        print(f"Failed to read context from {args.context_path}")
        return

    # Initialize and run the evaluator
    print(
        f"Initializing evaluator with gru model: {args.gru_model} and minion model: {args.minion_model}"
    )
    evaluator = Evaluator(gru_model=args.gru_model, minion_model=args.minion_model)

    print(f"Running evaluation for task: {args.task}")
    final_report = evaluator.run_evaluation(args.task, context)

    # Print the final report in a formatted way
    print("\n" + "=" * 50)
    print("FINAL SKILL EVALUATION REPORT")
    print("=" * 50)
    print(f"Task: {final_report['task']}")
    print(f"Overall Score: {final_report['overall_score']}/5.0")
    print(f"Overall Assessment: {final_report['overall_assessment']}")
    print("\nSkill Breakdown:")

    for i, skill_report in enumerate(final_report["skill_reports"]):
        print(f"\n{i+1}. {skill_report['skill_name']}")
        print(f"   Score: {skill_report['average_score']}/5.0")
        print(f"   Proficiency: {skill_report['proficiency_level']}")

        if skill_report["strengths"]:
            print("   Strengths:")
            for strength in skill_report["strengths"]:
                if strength.strip():
                    print(f"   - {strength}")

        if skill_report["weaknesses"]:
            print("   Weaknesses:")
            for weakness in skill_report["weaknesses"]:
                if weakness.strip():
                    print(f"   - {weakness}")

    # Save the report as JSON
    with open(args.output, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nDetailed report saved to '{args.output}'")


if __name__ == "__main__":
    main()


# sample usage
# python minion_evaluator.py --context_path "context.txt" --task "task.txt" --gru_model "gpt-4o-mini" --minion_model "llama3.2:1b" --output "evaluation_report.json"
