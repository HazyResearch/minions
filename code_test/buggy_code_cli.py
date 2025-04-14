#!/usr/bin/env python3
import argparse
import os
import json
import sys
import tempfile
import traceback
from typing import Dict, Any, List, Tuple
import openai
from processor import BuggyCodeProcessor

class BuggyCodeCLI:
    """
    CLI version of the Buggy Code Processor that works with a single Python file
    containing a class with buggy methods.
    """
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser for the CLI."""
        parser = argparse.ArgumentParser(
            description="Process a Python file containing a class with buggy methods, "
                        "extract functions, generate test cases, and fix bugs."
        )
        
        parser.add_argument(
            "input_file",
            help="Path to the Python file containing the buggy class"
        )
        
        parser.add_argument(
            "--output-dir", "-o",
            default="output",
            help="Directory to store the output files (default: 'output')"
        )
        
        parser.add_argument(
            "--threshold", "-t",
            type=float,
            default=0.7,
            help="Pass rate threshold (0.0-1.0) to consider a function fixed (default: 0.7)"
        )
        
        parser.add_argument(
            "--max-iterations", "-m",
            type=int,
            default=3,
            help="Maximum number of improvement attempts (default: 3)"
        )
        
        parser.add_argument(
            "--api-key", "-k",
            help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        
        return parser
    
    def _validate_input_file(self, file_path: str) -> bool:
        """Validate that the input file exists and is a Python file."""
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return False
        
        if not file_path.endswith(".py"):
            print(f"Warning: File does not have a .py extension: {file_path}")
            response = input("Continue anyway? (y/n): ")
            return response.lower() in ["y", "yes"]
        
        return True
    
    def _setup_api_key(self, api_key: str) -> bool:
        """Set up the OpenAI API key, either from args or environment variable."""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return True
        
        if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
            return True
        
        print("Error: OpenAI API key is required.")
        print("Either provide it with --api-key or set the OPENAI_API_KEY environment variable.")
        return False
    
    def _create_temp_csv(self, input_file: str) -> str:
        """
        Create a temporary CSV file that the processor can use.
        This adapts the input Python file to match the CSV format expected by the processor.
        """
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp_file:
            # Read the Python file content
            with open(input_file, "r") as py_file:
                python_content = py_file.read()
            
            # Create a CSV with headers matching what the processor expects
            tmp_file.write("File,Buggy Code,Errors,Buggy Functions\n")
            
            # Add the Python content as a row in the CSV
            csv_row = f'"{os.path.basename(input_file)}","{python_content.replace('"', '""')}","",""'
            tmp_file.write(csv_row)
            
            return tmp_file.name
    
    def _print_summary(self, results: Dict[str, Any], verbose: bool = False) -> None:
        """Print a summary of the processing results to the console."""
        print("\n===== PROCESSING SUMMARY =====")
        for func_name, data in results.items():
            # Calculate pass rate percentage for display
            pass_rate = data.get('pass_rate', 0)
            pass_percent = pass_rate * 100
            
            # Print basic function info
            print(f"\n{func_name}: {data['tests_passed']}/{data['tests_total']} tests passed ({pass_percent:.2f}%)")
            print(f"  Fixed by: {data['fixed_by']}")
            
            if data['fixed_by'] != 'original':
                print(f"  Iterations used: {data['iterations_used']}")
                
                # Show iteration progression
                if 'iterations_results' in data and verbose:
                    print("  Iteration progress:")
                    for iter_result in data['iterations_results']:
                        iter_pass_rate = iter_result['pass_rate'] * 100
                        print(f"    Iter {iter_result['iteration']}: "
                              f"{iter_result['passed']}/{iter_result['total']} ({iter_pass_rate:.2f}%)")
            
            # Show failures if any remain and verbose is enabled
            if data['failure_messages'] and verbose:
                print(f"  Remaining failures: {len(data['failure_messages'])}")
                for i, failure in enumerate(data['failure_messages'][:3]):  # Show first 3 failures
                    print(f"    Failure {i+1}: {failure[:100]}...")  # Truncate long messages
                if len(data['failure_messages']) > 3:
                    print(f"    ... and {len(data['failure_messages']) - 3} more failures")
    
    def run(self) -> int:
        """Run the CLI application with the provided arguments."""
        args = self.parser.parse_args()
        
        # Validate input file
        if not self._validate_input_file(args.input_file):
            return 1
        
        # Set up API key
        if not self._setup_api_key(args.api_key):
            return 1
        
        try:
            # Create temp CSV from Python file
            csv_path = self._create_temp_csv(args.input_file)
            
            print(f"Processing file: {args.input_file}")
            print(f"Output directory: {args.output_dir}")
            print(f"Pass threshold: {args.threshold}")
            print(f"Max iterations: {args.max_iterations}")
            print("Starting process...")
            
            # Create and run the processor
            processor = BuggyCodeProcessor(
                csv_path=csv_path,
                output_dir=args.output_dir,
                pass_threshold=args.threshold,
                max_iterations=args.max_iterations
            )
            
            # Process the code
            results = processor.process_all()
            
            # Print results summary
            self._print_summary(results, args.verbose)
            
            # Indicate where to find the full results
            print(f"\nDetailed results and fixed code saved to: {args.output_dir}")
            print(f"Summary available in: {os.path.join(args.output_dir, 'summary.json')}")
            
            # Cleanup temp file
            os.unlink(csv_path)
            
            return 0
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if args.verbose:
                print(traceback.format_exc())
            return 1


if __name__ == "__main__":
    cli = BuggyCodeCLI()
    sys.exit(cli.run()) 