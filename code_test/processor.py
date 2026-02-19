import pandas as pd
import ast
import astor
import subprocess
import json
import os
import openai
from typing import List, Dict, Tuple, Any, Union
import tempfile
import sys
import re
from ollama import chat
from pydantic import BaseModel

openai_key = os.getenv("OPENAI_API_KEY")

class BuggyCodeProcessor:
    def __init__(self, csv_path: str, output_dir: str = "processed_output", pass_threshold: float = 0.7, max_iterations: int = 3):
        """
        Initialize the processor with paths.
        
        Args:
            csv_path: Path to the CSV file containing buggy code
            output_dir: Directory to store output files
            pass_threshold: Threshold for test pass rate (0.0-1.0)
            max_iterations: Maximum number of attempts to fix a function
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.dataframe = None
        self.pass_threshold = pass_threshold
        self.max_iterations = max_iterations
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self) -> None:
        """Load the CSV data into a pandas DataFrame."""
        self.dataframe = pd.read_csv(self.csv_path)
        print(f"Loaded data with {len(self.dataframe)} rows")
        
    def extract_functions(self, code: str) -> Dict[str, str]:
        """
        Extract individual functions from a class definition using AST.
        
        Args:
            code: The complete code string
            
        Returns:
            Dictionary mapping function names to function code
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Find the import statements
            imports = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(astor.to_source(node).strip())
            
            # Prepare import string
            import_str = '\n'.join(imports) + '\n\n'
            
            # Find the class definition
            class_def = None
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_def = node
                    break
                    
            if not class_def:
                raise ValueError("No class definition found in code")
            
            # Extract methods from the class
            functions = {}
            for method in class_def.body:
                if isinstance(method, ast.FunctionDef):
                    method_name = method.name
                    
                    # create a copy of the method's arguments without 'self'
                    new_args = method.args
                    if new_args.args and new_args.args[0].arg == "self":
                        new_args.args.pop(0)

                    # Create a new module with just the function as a standalone function
                    new_func = ast.FunctionDef(
                        name=method.name,
                        args=method.args,
                        body=method.body,
                        decorator_list=[],
                        returns=method.returns
                    )
                    
                    # Create a new module and add the function
                    new_module = ast.Module(body=[new_func], type_ignores=[])
                    
                    # Convert back to source code
                    method_source = astor.to_source(new_module)
                    
                    # Add imports and store the function
                    functions[method_name] = import_str + method_source
            
            return functions
            
        except Exception as e:
            print(f"Error parsing code with AST: {e}")
            # Fallback to a simpler extraction if AST parsing fails
            print("Using fallback function extraction")
            return self._extract_functions_fallback(code)
    
    def _extract_functions_fallback(self, code: str) -> Dict[str, str]:
        """
        Fallback method to extract functions using simple string operations.
        This is less robust but serves as a backup if AST parsing fails.
        """
        import re
        
        # Extract the class name
        class_match = re.search(r'class\s+(\w+)', code)
        if not class_match:
            raise ValueError("No class definition found in code")
        
        # Find all function definitions within the class
        function_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*->[^:]*:\s*(?:"""[^"]*""")?\s*((?:.+?\n)+?)(?=\s+def|\s*$)'
        matches = re.finditer(function_pattern, code, re.MULTILINE | re.DOTALL)
        
        functions = {}
        imports = re.findall(r'import.*?$|from.*?import.*?$', code, re.MULTILINE)
        import_str = '\n'.join(imports) + '\n\n'
        
        for match in matches:
            func_name = match.group(1)
            func_body = match.group(2)

            # Remove 'self' parameter
            params = re.search(r'\(\s*(.*?)\s*\)', match.group(0)).group(1)
            params_list = [p.strip() for p in params.split(',')]
            filtered_params = [p for p in params_list if p and not p.startswith('self')]
            clean_params = ', '.join(filtered_params)
            
            # Create a standalone function (not method) with the imports
            standalone_func = import_str + f"def {func_name}({clean_params}):{func_body.rstrip()}"
            functions[func_name] = standalone_func
            
        return functions
    
    def create_func_files(self, func_name: str, func_code: str, suffix: str = ""):
        """
        Create .py files named func_name{suffix}.py containing each function and its corresponding code
        """
        # Write function code
        func_file_path = os.path.join(self.output_dir, f"{func_name}{suffix}.py")
        with open(func_file_path, "w") as f:
            f.write(func_code)

    def generate_json_test_cases(self, func_name: str, func_code: str) -> Dict:
        """
        Generate test cases in JSON format for accuracy evaluation using Qwen2.5-coder.
        
        Args:
            func_name: Name of the function
            func_code: Code of the function
            
        Returns:
            Dictionary with test cases data
        """

        class StructuredLocalOutput(BaseModel):
            test_cases: List[Dict[str, Any]]
        
        prompt = f"""Please generate 10 test cases for the following Python function. 
        Return ONLY a valid JSON object with the format shown below. Do not include any explanations or markdown formatting in your response.

        ```python
        {func_code}
        ```

        Format example:
        {{
            "test_cases": [
                {{
                    "name": "test_basic_case",
                    "arguments": ["parameter1", "parameter2"],
                    "correct": "output_for_correct_function"
                }},
                ...
            ]
        }}

        Make sure the input format matches the function parameters, and the output matches the return type."""

        try:
            response = chat(
                model='qwen2.5-coder',
                messages=[{'role': 'user', 'content': prompt}],
                format=StructuredLocalOutput.model_json_schema(),
            )
            
            # Parse the response to extract JSON
            structured_output = StructuredLocalOutput.model_validate_json(response.message.content)
            return structured_output.model_dump()
            
        except Exception as e:
            print(f"Error generating JSON test cases for {func_name}: {e}")
            # Return a minimal test case structure if there's an error
            return {"test_cases": [{"name": "test_dummy", "arguments": [], "correct": None}]}
    
    def generate_unittest_code(self, func_name: str, json_test_cases: Dict) -> str:
        """
        Generate unittest code from JSON test cases.
        
        Args:
            func_name: Name of the function
            json_test_cases: Dictionary with test cases
            
        Returns:
            String with unittest code
        """
        # Import the function module
        import_line = f"from {func_name} import {func_name}"
        
        # Create the TestCase class
        class_def = f"""
import unittest
import json

{import_line}

class Test{func_name.capitalize()}(unittest.TestCase):
"""
        
        # Add test methods
        test_methods = []
        for i, test_case in enumerate(json_test_cases.get("test_cases", [])):
            test_name = test_case.get("name", f"test_{i}")
            if not test_name.startswith("test_"):
                test_name = f"test_{test_name}"
            test_name = re.sub(r'\W', '_', test_name)  # Replace non-alphanumeric chars
            
            inputs = test_case.get("arguments", [])
            expected = test_case.get("correct", None)
            
            # Format the test method
            test_method = f"""
    def {test_name}(self):
        # Test case: {json.dumps(test_case)}
"""
            
            # Add the function call and assertion
            if isinstance(inputs, list):
                inputs_str = ", ".join(repr(inp) for inp in inputs)
                test_method += f"        result = {func_name}({inputs_str})\n"
            elif isinstance(inputs, dict):
                inputs_str = ", ".join(f"{k}={repr(v)}" for k, v in inputs.items())
                test_method += f"        result = {func_name}({inputs_str})\n"
            else:
                test_method += f"        result = {func_name}({repr(inputs)})\n"
            
            # Add appropriate assertion
            test_method += f"        self.assertEqual(result, {repr(expected)})\n"
            
            test_methods.append(test_method)
        
        # Add the main block
        main_block = """

if __name__ == "__main__":
    unittest.main()
"""
        
        # Combine all parts
        unittest_code = class_def + "".join(test_methods) + main_block
        
        return unittest_code
    
    def run_unit_tests(self, func_name: str, func_code: str, test_code: str) -> Tuple[int, int, List[Dict]]:
        """
        Run the unit tests and return the pass/fail statistics with detailed failure information.
        
        Args:
            func_name: Name of the function
            func_code: Code of the function
            test_code: Unit test code
            
        Returns:
            Tuple of (passed tests, total tests, test results)
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Write function to file
            func_file_path = os.path.join(temp_dir, f"{func_name}.py")
            with open(func_file_path, "w") as f:
                f.write(func_code)
            
            # Write test to file
            test_file_path = os.path.join(temp_dir, f"test_{func_name}.py")
            with open(test_file_path, "w") as f:
                f.write(test_code)
            
            # Run the tests
            result = subprocess.run(
                [sys.executable, test_file_path],
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            
            # Parse the output
            passed = 0
            total = 0
            test_results = []
            full_output = result.stdout + "\n" + result.stderr
            
            # First, try to parse total tests
            total_match = re.search(r'Ran (\d+) test', full_output)
            if total_match:
                total = int(total_match.group(1))
            
            # Check for failures
            if "OK" in full_output:
                passed = total
            else:
                # Count failures
                failures_match = re.search(r'FAILED \(([^)]+)\)', full_output)
                if failures_match:
                    failures_str = failures_match.group(1)
                    failures = 0
                    
                    # Parse failures and errors
                    if "failures=" in failures_str:
                        failures_count_match = re.search(r'failures=(\d+)', failures_str)
                        if failures_count_match:
                            failures += int(failures_count_match.group(1))
                    
                    if "errors=" in failures_str:
                        errors_count_match = re.search(r'errors=(\d+)', failures_str)
                        if errors_count_match:
                            failures += int(errors_count_match.group(1))
                    
                    passed = total - failures
            
            # Extract test cases from the test code
            test_case_pattern = r'# Test case: ({.*?})'
            test_case_matches = re.finditer(test_case_pattern, test_code, re.DOTALL)
            
            # Parse test cases from test code
            test_cases = []
            for match in test_case_matches:
                try:
                    test_case = json.loads(match.group(1))
                    test_cases.append(test_case)
                except Exception as e:
                    print(f"Error parsing test case JSON: {e}")
            
            # Extract full failure information
            # Look for patterns like: FAIL: test_name or ERROR: test_name
            failure_pattern = r'((?:FAIL|ERROR): test_\w+.*?(?=(?:FAIL|ERROR): test_|\Z))'
            failure_matches = re.finditer(failure_pattern, full_output, re.DOTALL)
            
            # Store all complete failure blocks
            failure_blocks = []
            for match in failure_matches:
                failure_blocks.append(match.group(1).strip())
            
            # Process each test case
            for test_case in test_cases:
                test_name = test_case.get("name", "unknown")
                if not test_name.startswith("test_"):
                    test_name = f"test_{test_name}"
                test_name = re.sub(r'\W', '_', test_name)  # Replace non-alphanumeric chars
                
                # Find if this test failed
                test_failure = None
                for failure in failure_blocks:
                    if test_name in failure:
                        test_failure = failure
                        break
                
                # Add to test results
                test_results.append({
                    "name": test_case.get("name", "unknown"),
                    "arguments": test_case.get("arguments", []),
                    "correct": test_case.get("correct", None),
                    "actual": None if test_failure else test_case.get("correct", None),
                    "passed": test_failure is None,
                    "error": test_failure if test_failure else ""
                })
            
            # If no test results were found, use a generic entry with all failure information
            if not test_results and total > 0:
                test_results.append({
                    "name": "generic_test",
                    "arguments": [],
                    "correct": None,
                    "actual": None,
                    "passed": passed == total,
                    "error": full_output if passed < total else ""
                })
            
            return passed, total, test_results
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
    
    def fix_with_gpt4o(self, api_key: str, func_name: str, func_code: str, error_messages: List[str], test_results: List[Dict], iteration: int = 1) -> str:
        """
        Get a fixed version of the function using GPT-4o.
        
        Args:
            api_key: OpenAI API key
            func_name: Name of the function
            func_code: Original buggy code
            error_messages: Error messages from failed tests
            test_results: Complete test results with detailed error information
            iteration: Current iteration number
            
        Returns:
            Fixed function code
        """
        print(f"Sending {func_name} to GPT-4o for fixing (iteration {iteration})...")
        
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare detailed test case information
        failed_tests_info = []
        for result in test_results:
            if not result.get("passed", True):
                test_info = (
                    f"Test: {result.get('name', 'unknown')}\n"
                    f"Arguments: {result.get('arguments', [])}\n"
                    f"Expected output: {result.get('correct', None)}\n"
                    f"Error: {result.get('error', '')}\n"
                )
                failed_tests_info.append(test_info)
        
        # Combine all failure information
        detailed_error_info = "\n".join(failed_tests_info)
        
        # Modify the prompt based on the iteration
        if iteration == 1:
            context = "This function has bugs that need to be fixed:"
        else:
            context = f"This function still has bugs after {iteration-1} attempt(s) to fix it:"
        
        prompt = f"""
        {context}
        ```python
        {func_code}
        ```
        
        The following unit tests are failing:
        
        {detailed_error_info}
        
        Please provide a corrected version of the function that will pass these tests.
        Return only the fixed code, without any explanations. Make sure to include imports!
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python expert who fixes bugs in code."},
                {"role": "user", "content": prompt}
            ]
        )

        fixed_code = response.choices[0].message.content.strip()
        # Remove Markdown code block formatting if present
        fixed_code = re.sub(r'^```python\s*\n', '', fixed_code)
        fixed_code = re.sub(r'^```python\s*', '', fixed_code)
        fixed_code = re.sub(r'\n```\s*$', '', fixed_code)
        fixed_code = re.sub(r'```\s*$', '', fixed_code)
        
        return fixed_code

    def process_all(self) -> Dict[str, Any]:
        """
        Process all code in the CSV and return results.
        
        Returns:
            Dictionary with results for each function
        """
        if self.dataframe is None:
            self.load_data()
        
        results = {}
        
        for _, row in self.dataframe.iterrows():
            buggy_code = row["Buggy Code"]
            
            # Extract individual functions
            functions = self.extract_functions(buggy_code)
            
            # Process each function
            for func_name, func_code in functions.items():
                print(f"\nProcessing function: {func_name}")
                
                # Create function file
                self.create_func_files(func_name, func_code, "_original")

                # Generate JSON test cases for accuracy evaluation
                print(f"Generating JSON test cases for {func_name} accuracy evaluation...")
                json_test_cases = self.generate_json_test_cases(func_name, func_code)
                
                # Generate unittest code from JSON test cases
                test_code = self.generate_unittest_code(func_name, json_test_cases)
                
                # Write test code to file
                test_file_path = os.path.join(self.output_dir, f"{func_name}_test.py")
                with open(test_file_path, "w") as f:
                    f.write(test_code)
                
                # Run unit tests on the original code
                print(f"Running unit tests for {func_name}...")
                passed, total, test_results = self.run_unit_tests(func_name, func_code, test_code)
                print(f"Test results: {passed}/{total} passed")
                
                # Calculate initial pass rate
                pass_rate = float(passed) / float(total) if total > 0 else 0.0
                current_code = func_code
                
                # Start with the original code as "fixed_by"
                fixed_by = "original"
                
                # Only attempt fixes if we're below the threshold
                iterations_used = 0
                all_iteration_results = [{
                    "iteration": 0,
                    "pass_rate": pass_rate,
                    "passed": passed,
                    "total": total,
                    "code": current_code
                }]
                
                # Initialize failure messages
                failure_messages = [result.get("error", "") for result in test_results if not result.get("passed", False)]
                
                # Iterative improvement if needed
                if pass_rate < self.pass_threshold:
                    fixed_by = "gpt4o"
                    for iteration in range(1, self.max_iterations + 1):
                        iterations_used = iteration
                        
                        # Get fixed code from GPT-4o
                        print(f"Pass rate {pass_rate:.2f} < {self.pass_threshold}, Iteration {iteration}/{self.max_iterations}")
                        fixed_code = self.fix_with_gpt4o(
                            openai_key, 
                            func_name, 
                            current_code, 
                            failure_messages, 
                            test_results,
                            iteration
                        )
                        
                        # Write fixed code to file
                        self.create_func_files(func_name, fixed_code, f"_iter_{iteration}")
                        
                        # Test the fixed code
                        print(f"Testing iteration {iteration} for {func_name}...")
                        fixed_passed, fixed_total, fixed_test_results = self.run_unit_tests(
                            func_name, fixed_code, test_code
                        )
                        print(f"Iteration {iteration} results: {fixed_passed}/{fixed_total} passed")
                        
                        # Update pass rate
                        new_pass_rate = float(fixed_passed) / float(fixed_total) if fixed_total > 0 else 0.0
                        
                        # Store iteration results
                        all_iteration_results.append({
                            "iteration": iteration,
                            "pass_rate": new_pass_rate,
                            "passed": fixed_passed,
                            "total": fixed_total,
                            "code": fixed_code
                        })
                        
                        # Update current state
                        current_code = fixed_code
                        passed = fixed_passed
                        total = fixed_total
                        pass_rate = new_pass_rate
                        test_results = fixed_test_results
                        failure_messages = [result.get("error", "") for result in fixed_test_results if not result.get("passed", False)]
                        
                        # Break if we've reached the threshold
                        if pass_rate >= self.pass_threshold:
                            print(f"Success! Reached pass rate {pass_rate:.2f} >= {self.pass_threshold}")
                            break
                
                # Select the best iteration based on pass rate
                best_iteration = max(all_iteration_results, key=lambda x: x["pass_rate"])
                final_code = best_iteration["code"]
                final_pass_rate = best_iteration["pass_rate"]
                final_passed = best_iteration["passed"]
                final_total = best_iteration["total"]
                
                # Log which iteration was best
                if best_iteration["iteration"] > 0:
                    print(f"Best result was from iteration {best_iteration['iteration']} with pass rate {final_pass_rate:.2f}")
                else:
                    print(f"Original code had best pass rate: {final_pass_rate:.2f}")
                
                # Save the final code
                self.create_func_files(func_name, final_code, "_final")
                
                # Store results
                results[func_name] = {
                    "original_code": func_code,
                    "final_code": final_code,
                    "tests_passed": final_passed,
                    "tests_total": final_total,
                    "pass_rate": final_pass_rate,
                    "fixed_by": fixed_by,
                    "iterations_used": iterations_used,
                    "iterations_results": all_iteration_results,
                    "failure_messages": failure_messages,
                    "test_results": test_results,
                }
        
        # Write summary to file
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            # Create a copy without the large test_results field to keep the summary manageable
            summary_results = {}
            for func_name, data in results.items():
                summary_results[func_name] = {
                    k: v for k, v in data.items() 
                    if k not in ['test_results', 'detailed_results', 'iterations_results']
                }
                # Add a simplified version of iterations_results
                if 'iterations_results' in data:
                    summary_results[func_name]['iterations_summary'] = [
                        {
                            'iteration': r['iteration'],
                            'pass_rate': r['pass_rate'],
                            'passed': r['passed'],
                            'total': r['total']
                        }
                        for r in data['iterations_results']
                    ]
            
            json.dump(summary_results, f, indent=4)
            
        return results

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process buggy code CSV')
    parser.add_argument('--csv', default='buggy_code.csv', help='Path to the CSV file')
    parser.add_argument('--output', default='processed_output', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.7, help='Pass rate threshold (0.0-1.0)')
    parser.add_argument('--max-iterations', type=int, default=3, help='Maximum fix iterations')
    args = parser.parse_args()
    
    processor = BuggyCodeProcessor(args.csv, args.output, args.threshold, args.max_iterations)
    results = processor.process_all()
    
    # Print summary
    print("\n===== PROCESSING SUMMARY =====")
    for func_name, data in results.items():
        print(f"{func_name}: {data['tests_passed']}/{data['tests_total']} tests passed ({data['pass_rate']:.2f})")
        print(f"  Fixed by: {data['fixed_by']}")
        if data['fixed_by'] != 'original':
            print(f"  Iterations used: {data['iterations_used']}")
            
            # Show iteration progression if available
            if 'iterations_results' in data:
                print("  Iteration progress:")
                for iter_result in data['iterations_results']:
                    print(f"    Iter {iter_result['iteration']}: {iter_result['passed']}/{iter_result['total']} ({iter_result['pass_rate']:.2f})")
                    
        if data['failure_messages']:
            print(f"  Remaining failures: {len(data['failure_messages'])}")
    
if __name__ == "__main__":
    main()