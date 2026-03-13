import unittest
from ollama import Client
import json
from typing import List

class TestQwenTestGenerator(unittest.TestCase):
    def fibonacci(self, n: int) -> List[int]:
        """Generate the first n Fibonacci numbers."""
        fib = [0, 1]
        for _ in range(n - 2):
            fib.append(fib[-1] - fib[-2])
        return fib[:n]

    def test_generate_fibonacci_tests(self):
        # Initialize Ollama client
        client = Client()
        
        # Function code to test
        function_code = '''
def fibonacci(self, n: int) ->List[int]:
    """Generate the first n Fibonacci numbers."""
    fib = [0, 1]
    for _ in range(n - 2):
        fib.append(fib[-1] - fib[-2])
    return fib[:n]
'''

        # Prompt for Qwen to generate test cases
        prompt = f"""Please generate 10 test cases for the following Python function. 
Return them in a JSON format with test name and test inputs/expected outputs.

{function_code}

Format example:
{{
    "test_cases": [
        {{
            "name": "test_basic_case",
            "input": 5,
            "expected": [0, 1, 1, 2, 3]
        }},
        ...
    ]
}}"""

        # Get response from Qwen
        response = client.chat(model='qwen2.5-coder', messages=[{
            'role': 'user',
            'content': prompt
        }])

        # Parse the response to extract JSON
        response_text = response['message']['content']
        json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
        test_cases = json.loads(json_str)

        # Print the test cases for inspection
        print("\nGenerated test cases:")
        print(json.dumps(test_cases, indent=2))

        # Run the test cases and calculate accuracy
        correct_tests = 0
        total_tests = len(test_cases['test_cases'])

        for test_case in test_cases['test_cases']:
            input_n = test_case['input']
            expected = test_case['expected']
            actual = self.fibonacci(input_n)
            if actual == expected:
                correct_tests += 1
            print(f"\nTest: {test_case['name']}")
            print(f"Input: {input_n}")
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            print(f"Pass: {actual == expected}")

        accuracy = (correct_tests / total_tests) * 100
        print(f"\nAccuracy: {accuracy}% ({correct_tests}/{total_tests} tests passed)")

if __name__ == '__main__':
    unittest.main()
