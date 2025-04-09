import os
import time
from typing import List, Dict
from minions.utils.minion_mcp import _make_mcp_minion
import PyPDF2

class MCPToolEvaluator:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.minion = _make_mcp_minion("filesystem")
        self.mcp_tool_calls = []
        
    def _track_tool_call(self, tool_name: str, params: Dict):
        """Track MCP tool calls for evaluation"""
        self.mcp_tool_calls.append({
            "tool": tool_name,
            "params": params,
            "timestamp": time.time()
        })
        
    def _ensure_text_file(self, file_path: str) -> str:
        """
        Ensures a text version of the file exists. If not, converts from PDF.
        Returns the path to the text file.
        """
        # Check if text version exists
        text_path = file_path.replace('.pdf', '.txt')
        if os.path.exists(text_path):
            print(f"Using existing text file: {text_path}")
            return text_path
            
        # If PDF exists, convert it
        if os.path.exists(file_path) and file_path.endswith('.pdf'):
            try:
                print(f"Converting PDF to text: {file_path}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(text_path), exist_ok=True)
                
                # Read PDF
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                
                # Write text file
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text_content)
                
                print(f"Successfully created text file: {text_path}")
                return text_path
            except Exception as e:
                print(f"Error converting PDF to text: {e}")
                return file_path
        else:
            print(f"Neither PDF nor text file found: {file_path}")
        
        return file_path
        
    def evaluate_tool_combination(self, 
                                document_paths: List[str], 
                                query: str, 
                                tools: List[str]) -> Dict:
        """
        Evaluate a combination of MCP tools on given documents and query.
        """
        try:
            # Reset tool call tracking
            self.mcp_tool_calls = []
            
            # Convert to absolute paths, ensure text versions exist, and check existence
            abs_paths = []
            for path in document_paths:
                abs_path = os.path.join(self.project_root, path)
                # Convert to text if needed
                text_path = self._ensure_text_file(abs_path)
                if not os.path.exists(text_path):
                    return {
                        "success": False,
                        "error": f"File not found and conversion failed: {path}",
                        "tools": tools,
                        "mcp_tool_calls": self.mcp_tool_calls
                    }
                abs_paths.append(text_path)
            
            # Run the minion with specific context about the files
            context = [
                "I have the following earnings reports to analyze:",
                *[f"{i+1}. {path} - {os.path.basename(path)}" for i, path in enumerate(abs_paths)]
            ]
            
            # Run the minion and track tool calls from the output
            output = self.minion(
                task=query,
                context=context,
                max_rounds=3,
                logging_id=int(time.time()),
            )
            
            # Extract tool calls from the output
            if 'tool_calls' in output:
                for call in output['tool_calls']:
                    self._track_tool_call(call.get('tool_name', ''), call.get('parameters', {}))
            
            # Basic evaluation metrics
            return {
                "success": True,
                "tools": tools,
                "documents": len(document_paths),
                "response_length": len(output.get('final_answer', '')),
                "tool_calls": len(output.get('tool_calls', [])),
                "mcp_tool_calls": self.mcp_tool_calls,
                "response": output.get('final_answer', ''),
                "tool_effectiveness": {
                    "correct_tools_used": any(call["tool"] in tools for call in self.mcp_tool_calls),
                    "achieved_goal": len(output.get('final_answer', '')) > 0,
                    "errors_encountered": len([call for call in self.mcp_tool_calls if "error" in str(call.get("params", {}))]) > 0
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tools": tools,
                "mcp_tool_calls": self.mcp_tool_calls
            }

def print_evaluation_summary(evaluations: List[Dict]):
    """Print a comprehensive and visually appealing summary of the evaluations."""
    print("\n" + "="*50)
    print("MCP TOOL EVALUATION SUMMARY".center(50))
    print("="*50)
    
    successful = [e for e in evaluations if e["success"]]
    failed = [e for e in evaluations if not e["success"]]
    
    # Print overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  • Total Evaluations: {len(evaluations)}")
    print(f"  • Successful: {len(successful)} ✅")
    print(f"  • Failed: {len(failed)} ❌")
    
    # Print successful evaluations
    if successful:
        print("\n" + "✅ SUCCESSFUL EVALUATIONS".center(50, "-"))
        for i, eval in enumerate(successful, 1):
            print(f"\nEvaluation #{i}:")
            print(f"  • Tools Configured: {', '.join(eval['tools'])}")
            print(f"  • Documents Processed: {eval['documents']}")
            
            # Tool usage analysis
            print("\n  🔧 Tool Usage Analysis:")
            mcp_calls = eval.get('mcp_tool_calls', [])
            print(f"    • Total MCP Tool Calls: {len(mcp_calls)}")
            if mcp_calls:
                print("    • Tools Actually Used:")
                for call in mcp_calls:
                    print(f"      - {call['tool']}")
                    print(f"        Parameters: {call['params']}")
            
            # Effectiveness analysis
            if "tool_effectiveness" in eval:
                print("\n  📈 Tool Effectiveness:")
                effectiveness = eval["tool_effectiveness"]
                print(f"    • Used Correct Tools: {'✅ Yes' if effectiveness['correct_tools_used'] else '❌ No'}")
                print(f"    • Achieved Goal: {'✅ Yes' if effectiveness['achieved_goal'] else '❌ No'}")
                print(f"    • Encountered Errors: {'❌ Yes' if effectiveness['errors_encountered'] else '✅ No'}")
            
            # Response analysis
            print("\n  📝 Response Analysis:")
            print(f"    • Length: {eval['response_length']} characters")
            if 'response' in eval:
                print("    • Preview:")
                preview = eval['response'][:200] + "..." if len(eval['response']) > 200 else eval['response']
                print(f"      {preview}")
    
    # Print failed evaluations
    if failed:
        print("\n" + "❌ FAILED EVALUATIONS".center(50, "-"))
        for i, eval in enumerate(failed, 1):
            print(f"\nEvaluation #{i}:")
            print(f"  • Tools: {', '.join(eval['tools'])}")
            print(f"  • Error: {eval['error']}")
            if eval.get('mcp_tool_calls'):
                print(f"  • Tool Calls Before Failure: {len(eval['mcp_tool_calls'])}")
                print("    Tools Used:")
                for call in eval['mcp_tool_calls']:
                    print(f"      - {call['tool']}")
                    print(f"        Parameters: {call['params']}")
    
    print("\n" + "="*50)
    print("END OF EVALUATION".center(50))
    print("="*50 + "\n")

def main():
    # Test case
    test_case = {
        "documents": [
            "minions/examples/finance/pdfs/PEPSICO_2023Q2_EARNINGS.pdf",
            "minions/examples/finance/pdfs/SALESFORCE_2024Q2_EARNINGS.pdf",
            "minions/examples/finance/pdfs/ULTABEAUTY_2024Q2_EARNINGS.pdf"
        ],
        "query": """
        Please analyze earnings trends by comparing:
        1. PepsiCo's 2023 Q2 earnings
        2. Salesforce's 2024 Q2 earnings
        3. Ulta Beauty's 2024 Q2 earnings
        
        For each company, provide:
        1. Revenue growth vs previous quarter
        2. Key drivers of performance
        3. Forward guidance
        4. Market reaction
        """,
        "tools": ["read_multiple_files"]
    }
    
    # Set up and run evaluation
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    evaluator = MCPToolEvaluator(project_root)
    
    evaluation = evaluator.evaluate_tool_combination(
        test_case["documents"],
        test_case["query"],
        test_case["tools"]
    )
    
    # Print summary
    print_evaluation_summary([evaluation])

if __name__ == "__main__":
    main() 