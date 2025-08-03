import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from minions.usage import Usage
from minions.utils.workspace import WorkspaceManager
from minions.clients import OpenAIClient, TogetherClient, GeminiClient

from minions.prompts.minion_code import (
    RUNBOOK_GENERATION_PROMPT,
    SUBTASK_EXECUTION_PROMPT,
    CODE_REVIEW_PROMPT,
    EDIT_REQUEST_PROMPT,
    FINAL_INTEGRATION_PROMPT,
)

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    import re
    
    # First try to find JSON in markdown code blocks
    block_matches = list(re.finditer(r"``````", text, re.DOTALL))
    if block_matches:
        json_str = block_matches[-1].group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Fallback: Extract complete JSON object with balanced braces
    json_str = _extract_balanced_json(text)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        raise

def _extract_balanced_json(text: str) -> str:
    """Extract a complete JSON object by counting balanced braces."""
    start_idx = text.find('{')
    if start_idx == -1:
        return text
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
    
    # If we get here, braces weren't balanced - return original text
    return text



class DevMinion:
    """Development-focused minion that creates software projects through supervised local development."""
    
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        workspace_dir: str = "dev_workspace",
        max_edit_rounds: int = 3,
        callback=None,
        log_dir: str = "dev_minion_logs",
        backup_enabled: bool = True,
    ):
        """
        Initialize the DevMinion.
        
        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            workspace_dir: Directory where development work will happen
            max_edit_rounds: Maximum rounds of edits per subtask
            callback: Optional callback function for progress updates
            log_dir: Directory for logging development session
            backup_enabled: Whether to create backups of workspace
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_edit_rounds = max_edit_rounds
        self.callback = callback
        self.log_dir = log_dir
        
        # Initialize workspace manager
        self.workspace = WorkspaceManager(workspace_dir, backup_enabled)
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Development session state
        self.runbook = None
        self.current_step = 0
        self.completed_steps = []
        self.step_documentation = {}
        
    def __call__(
        self,
        task: str,
        requirements: str = "",
        max_steps: Optional[int] = None,
        logging_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a development task using the DevMinion protocol.
        
        Args:
            task: The development task description
            requirements: Additional requirements or context
            max_steps: Maximum number of steps to execute (None = all steps)
            logging_id: Optional identifier for logging
            
        Returns:
            Dictionary containing project results and session information
        """
        print("\n========== DEV MINION TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Requirements: {requirements}")
        print(f"Workspace: {self.workspace.workspace_dir}")
        
        start_time = time.time()
        
        # Initialize session log
        session_log = {
            "task": task,
            "requirements": requirements,
            "start_time": datetime.now().isoformat(),
            "workspace_dir": str(self.workspace.workspace_dir),
            "runbook": None,
            "steps_completed": [],
            "final_assessment": None,
            "usage": {
                "remote": {},
                "local": {},
            },
            "timing": {
                "total_time": 0.0,
                "runbook_generation_time": 0.0,
                "development_time": 0.0,
                "review_time": 0.0,
            }
        }
        
        remote_usage = Usage()
        local_usage = Usage()
        
        try:
            # Step 1: Generate runbook
            print("\n=== GENERATING DEVELOPMENT RUNBOOK ===")
            if self.callback:
                self.callback("supervisor", "Generating development runbook...", is_final=False)
            
            runbook_start = time.time()
            runbook_result = self._generate_runbook(task, requirements)
            session_log["runbook"] = runbook_result["runbook"]
            remote_usage += runbook_result["usage"]
            session_log["timing"]["runbook_generation_time"] = time.time() - runbook_start
            
            # Step 2: Execute development steps
            print(f"\n=== EXECUTING {len(self.runbook['steps'])} DEVELOPMENT STEPS ===")
            dev_start = time.time()
            
            steps_to_execute = self.runbook["steps"]
            if max_steps:
                steps_to_execute = steps_to_execute[:max_steps]
            
            for step_info in steps_to_execute:
                step_result = self._execute_development_step(step_info)
                session_log["steps_completed"].append(step_result)
                local_usage += step_result["local_usage"]
                remote_usage += step_result["remote_usage"]
                
                if not step_result["success"]:
                    print(f"❌ Step {step_info['step_number']} failed. Stopping execution.")
                    break
            
            session_log["timing"]["development_time"] = time.time() - dev_start
            
            # Step 3: Final integration assessment
            print("\n=== CONDUCTING FINAL INTEGRATION REVIEW ===")
            if self.callback:
                self.callback("supervisor", "Conducting final integration review...", is_final=False)
            
            review_start = time.time()
            final_assessment = self._conduct_final_review(task)
            session_log["final_assessment"] = final_assessment["assessment"]
            remote_usage += final_assessment["usage"]
            session_log["timing"]["review_time"] = time.time() - review_start
            
        except Exception as e:
            self.logger.error(f"Development session failed: {e}")
            session_log["error"] = str(e)
        
        # Finalize session log
        session_log["timing"]["total_time"] = time.time() - start_time
        session_log["usage"]["remote"] = remote_usage.to_dict()
        session_log["usage"]["local"] = local_usage.to_dict()
        session_log["end_time"] = datetime.now().isoformat()
        
        # Save session log
        log_filename = self._save_session_log(session_log, logging_id)
        
        print(f"\n=== DEV MINION SESSION COMPLETED ===")
        print(f"Log saved to: {log_filename}")
        
        # Generate final summary
        workspace_summary = self.workspace.get_project_summary()
        
        return {
            "success": session_log.get("error") is None,
            "runbook": self.runbook,
            "steps_completed": len(session_log["steps_completed"]),
            "total_steps": len(self.runbook["steps"]) if self.runbook else 0,
            "final_assessment": session_log.get("final_assessment"),
            "workspace_dir": str(self.workspace.workspace_dir),
            "workspace_summary": workspace_summary,
            "session_log": session_log,
            "log_file": log_filename,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
        }
    
    def _generate_runbook(self, task: str, requirements: str) -> Dict[str, Any]:
        """Generate the development runbook using the remote LLM."""
        prompt = RUNBOOK_GENERATION_PROMPT.format(
            task=task,
            requirements=requirements or "No specific requirements provided"
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Call remote LLM for runbook generation
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            response, usage = self.remote_client.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
        elif isinstance(self.remote_client, GeminiClient):
            from pydantic import BaseModel
            from typing import List as TypeList, Dict as TypeDict
            
            class Tests(BaseModel):
                test_files: TypeDict[str, str]
                test_commands: TypeList[str]
                test_documentation: str
            
            class Step(BaseModel):
                step_number: int
                title: str
                description: str
                files_to_create: TypeList[str]
                files_to_modify: TypeList[str]
                tests: Tests
                acceptance_criteria: str
            
            class RunbookOutput(BaseModel):
                project_overview: str
                technology_stack: TypeList[str]
                steps: TypeList[Step]
                final_testing: str
            
            response, usage = self.remote_client.chat(
                messages=messages,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RunbookOutput,
                }
            )
        else:
            response, usage = self.remote_client.chat(messages=messages)
        
        # Parse runbook
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient, GeminiClient)):
            try:
                runbook_data = json.loads(response[0])
            except:
                runbook_data = _extract_json(response[0])
        else:
            runbook_data = _extract_json(response[0])
        
        self.runbook = runbook_data
        
        print(f"📋 Generated runbook with {len(runbook_data['steps'])} steps")
        print(f"🛠️  Technology stack: {', '.join(runbook_data['technology_stack'])}")
        
        if self.callback:
            callback_message = {
                "role": "assistant",
                "content": f"Generated development runbook:\n\n**Project:** {runbook_data['project_overview']}\n\n**Steps:** {len(runbook_data['steps'])}\n\n**Tech Stack:** {', '.join(runbook_data['technology_stack'])}"
            }
            self.callback("supervisor", callback_message, is_final=False)
        
        return {
            "runbook": runbook_data,
            "usage": usage
        }
    
    def _execute_development_step(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single development step with local LLM and remote review."""
        step_number = step_info["step_number"]
        step_title = step_info["title"]
        
        print(f"\n--- Step {step_number}: {step_title} ---")
        
        if self.callback:
            self.callback("worker", f"Starting step {step_number}: {step_title}", is_final=False)
        


        step_result = {
            "step_number": step_number,
            "step_title": step_title,
            "success": False,
            "attempts": [],
            "final_code_changes": {},
            "documentation": "",
            "test_results": {},
            "local_usage": Usage(),
            "remote_usage": Usage(),
        }
        
        # Create backup before starting step
        backup_path = self.workspace.create_backup(f"before_step_{step_number}")
        
        # Get current workspace state
        initial_state = self.workspace.get_current_state()
        
        # Track feedback from previous attempts
        previous_feedback = []
        
        for attempt in range(1, self.max_edit_rounds + 1):
            print(f"  Attempt {attempt}/{self.max_edit_rounds}")
            initial_state = self.workspace.get_current_state()
            
            attempt_result = {
                "attempt_number": attempt,
                "local_response": None,
                "code_changes": {},
                "test_generation": None,
                "test_results": {},
                "review_decision": None,
                "review_feedback": None,
            }
            
            # Local LLM implements the step (without tests)
            implementation_result = self._local_implement_step(step_info, initial_state, previous_feedback)
            attempt_result["local_response"] = implementation_result["response"]
            attempt_result["code_changes"] = implementation_result["files"]
            step_result["local_usage"] += implementation_result["usage"]
            
            # Apply code changes to workspace
            if implementation_result["files"]:
                changes_applied = self.workspace.apply_file_changes(implementation_result["files"])
                print(f"    Applied {len(changes_applied)} file changes")
            
            # Use predefined tests from runbook
            test_commands = []
            test_results = {}
            
            # Apply predefined test files from runbook
            if step_info.get("tests", {}).get("test_files"):
                test_files = step_info["tests"]["test_files"]
                # add a folder for the test files
                test_files_folder = self.workspace.workspace_dir / "test_files"
                # make sure the test_files_folder exists
                test_files_folder.mkdir(parents=True, exist_ok=True)
                
                test_changes_applied = self.workspace.apply_file_changes(test_files)
                print(f"    Applied {len(test_changes_applied)} predefined test files")
                
                # Run the predefined tests
                test_commands = step_info["tests"].get("test_commands", [])
                if test_commands:
                    test_results = self.workspace.run_tests(test_commands)
                    print(test_results)
                    attempt_result["test_results"] = test_results
                    print(f"    Test results: {test_results['summary']}")
            
            # Remote LLM reviews the implementation
            review_result = self._remote_review_step(step_info, implementation_result, test_results)
            attempt_result["review_decision"] = review_result["decision"]
            attempt_result["review_feedback"] = review_result["feedback"]
            step_result["remote_usage"] += review_result["usage"]
            
            step_result["attempts"].append(attempt_result)
            
            if review_result["decision"] == "merge_changes":
                print(f"  ✅ Step {step_number} approved and merged")
                step_result["success"] = True
                step_result["final_code_changes"] = implementation_result["files"]
                step_result["documentation"] = implementation_result["documentation"]
                step_result["test_results"] = test_results
                
                # Create documentation file
                if implementation_result["documentation"]:
                    doc_path = self.workspace.create_documentation_file(
                        step_number, 
                        implementation_result["documentation"]
                    )
                    print(f"    📝 Created documentation: {doc_path}")
                
                break
            else:
                print(f"  🔄 Step {step_number} needs revisions (attempt {attempt})")
                if self.callback:
                    self.callback("supervisor", f"Requesting revisions for step {step_number}: {review_result['feedback'].get('issues', [])}", is_final=False)
                
                # Add this attempt's feedback to the list for the next attempt
                previous_feedback.append({
                    "attempt_number": attempt,
                    "feedback": review_result["feedback"],
                    "issues": review_result["feedback"].get("issues", []),
                    "suggestions": review_result["feedback"].get("suggestions", [])
                })
                
                if attempt == self.max_edit_rounds:
                    print(f"  ❌ Step {step_number} failed after {self.max_edit_rounds} attempts")
                    break
        
        self.completed_steps.append(step_result)
        return step_result
    
    def _local_implement_step(self, step_info: Dict[str, Any], current_state: Dict[str, Any], previous_feedback: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Have local LLM implement a development step."""
        current_workspace = self._format_workspace_state(current_state)
        completed_steps_summary = self._format_completed_steps()
        
        # Format previous feedback section
        previous_feedback_section = ""
        if previous_feedback:
            previous_feedback_section = "**Previous Review Feedback:**\n"
            previous_feedback_section += "Learn from the following feedback from previous attempts to improve your implementation:\n\n"
            for feedback in previous_feedback:
                attempt_num = feedback["attempt_number"]
                issues = feedback.get("issues", [])
                suggestions = feedback.get("suggestions", [])
                
                previous_feedback_section += f"*Attempt {attempt_num} Feedback:*\n"
                if issues:
                    previous_feedback_section += f"- Issues found: {', '.join(issues)}\n"
                if suggestions:
                    previous_feedback_section += f"- Suggestions: {', '.join(suggestions)}\n"
                previous_feedback_section += "\n"
            
            previous_feedback_section += "Please address these issues and incorporate the suggestions in your current implementation.\n"
        
        # Format predefined tests for display
        predefined_tests = ""
        if step_info.get("tests", {}).get("test_files"):
            predefined_tests = "Test Files:\n"
            for test_file, test_content in step_info["tests"]["test_files"].items():
                predefined_tests += f"\n--- {test_file} ---\n{test_content}\n"
            predefined_tests += f"\nTest Commands: {step_info['tests'].get('test_commands', [])}\n"
            predefined_tests += f"Test Documentation: {step_info['tests'].get('test_documentation', 'No documentation provided')}\n"
        else:
            predefined_tests = "No tests defined for this step."


        files_to_create = step_info["files_to_create"]
        files_to_modify = step_info["files_to_modify"]

        if files_to_create:
            files_to_create = ", ".join(files_to_create)
        if files_to_modify:
            files_to_modify = ", ".join(files_to_modify)        
        # get files to create and modify
        if files_to_create:
            current_file_state = self.workspace.get_file_contents(files_to_create)
            # append that to the current_workspace
            current_workspace += f"\n\nCurrent File State:\n{files_to_create}:\n{current_file_state}"
        if files_to_modify:
            current_file_state = self.workspace.get_file_contents(files_to_modify)
            # append that to the current_workspace
            current_workspace += f"\n\nCurrent File State:\n{files_to_modify}:\n{current_file_state}"
        
        prompt = SUBTASK_EXECUTION_PROMPT.format(
            step_number=step_info["step_number"],
            step_title=step_info["title"],
            step_description=step_info["description"],
            files_to_create=", ".join(step_info["files_to_create"]),
            files_to_modify=", ".join(step_info["files_to_modify"]),
            predefined_tests=predefined_tests,
            acceptance_criteria=step_info.get("acceptance_criteria", "Default: Complete all requirements as described in this step"),
            current_workspace=current_workspace,
            completed_steps=completed_steps_summary,
            previous_feedback_section=previous_feedback_section,
        )


        messages = [{"role": "user", "content": prompt}]

        response, usage, _  = self.local_client.chat(messages=messages)
        # Parse the JSON response
        try:
            try:
                implementation_data = json.loads(response[0])
            except Exception as e:
                implementation_data = _extract_json(response[0])
        except Exception as e:
            # Fallback if JSON parsing fails
            implementation_data = {
                "files": {},
                "documentation": response[0],
                "setup_instructions": [],
                "completion_notes": "Implementation completed (JSON parsing failed)"
            }
        
        if self.callback:
            callback_message = {
                "role": "assistant", 
                "content": f"Step {step_info['step_number']} implementation:\n\n{implementation_data.get('completion_notes', 'Implementation completed')}"
            }
            self.callback("worker", callback_message, is_final=False)
        
        return {
            "response": response[0],
            "files": implementation_data.get("files", {}),
            "documentation": implementation_data.get("documentation", ""),
            "setup_instructions": implementation_data.get("setup_instructions", []),
            "completion_notes": implementation_data.get("completion_notes", ""),
            "usage": usage,
        }
    

    
    def _remote_review_step(self, step_info: Dict[str, Any], implementation: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Have remote LLM review the implementation."""
        current_state = self.workspace.get_current_state()
        workspace_summary = self._format_workspace_state(current_state)
        
        # Format code changes for review
        code_changes = ""
        for filepath, content in implementation["files"].items():
            code_changes += f"\n--- {filepath} ---\n{content}\n"


        
        prompt = CODE_REVIEW_PROMPT.format(
            step_number=step_info["step_number"],
            step_title=step_info["title"],
            acceptance_criteria=step_info.get("acceptance_criteria", "Default: Complete all requirements as described in this step"),
            code_changes=code_changes,
            documentation=implementation["documentation"],
            test_results=json.dumps(test_results, indent=2),
            workspace_state=workspace_summary,
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            response, usage = self.remote_client.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
        elif isinstance(self.remote_client, GeminiClient):
            from pydantic import BaseModel
            from typing import List as TypeList
            
            class Feedback(BaseModel):
                strengths: TypeList[str]
                issues: TypeList[str]
                suggestions: TypeList[str]
            
            class ReviewOutput(BaseModel):
                decision: str
                overall_score: str
                feedback: Feedback
                specific_changes_needed: TypeList[str]
                approval_notes: str
            
            response, usage = self.remote_client.chat(
                messages=messages,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ReviewOutput,
                },
                response_format={"type": "json_object"}
            )
        else:
            response, usage = self.remote_client.chat(messages=messages, response_format={"type": "json_object"})
        
        # Parse review decision
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient, GeminiClient)):
            try:
                review_data = json.loads(response[0])
            except:
                review_data = _extract_json(response[0])
        else:
            review_data = _extract_json(response[0])
        
        decision = review_data.get("decision", "request_edits")
        
        print(f"    🔍 Review decision: {decision}")
        print(f"    📊 Overall score: {review_data.get('overall_score', 'N/A')}")
        
        return {
            "decision": decision,
            "feedback": review_data.get("feedback", {}),
            "review_data": review_data,
            "usage": usage,
        }
    
    def _conduct_final_review(self, original_task: str) -> Dict[str, Any]:
        """Conduct final integration review of the completed project."""
        final_state = self.workspace.get_current_state()
        workspace_summary = self._format_workspace_state(final_state)
        
        # Gather all documentation
        all_docs = ""
        for step_result in self.completed_steps:
            if step_result.get("documentation"):
                all_docs += f"\n## Step {step_result['step_number']}\n{step_result['documentation']}\n"
        
        # Run final tests
        final_test_results = self.workspace.run_tests()
        
        prompt = FINAL_INTEGRATION_PROMPT.format(
            original_task=original_task,
            project_overview=self.runbook.get("project_overview", ""),
            completed_steps=len(self.completed_steps),
            final_workspace=workspace_summary,
            all_documentation=all_docs,
            final_test_results=json.dumps(final_test_results, indent=2),
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            response, usage = self.remote_client.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
        elif isinstance(self.remote_client, GeminiClient):
            from pydantic import BaseModel
            from typing import List as TypeList
            
            class Assessment(BaseModel):
                strengths: TypeList[str]
                weaknesses: TypeList[str]
                missing_features: TypeList[str]
                quality_score: str
            
            class DeploymentReadiness(BaseModel):
                ready_to_deploy: bool
                setup_instructions: str
                known_issues: TypeList[str]
            
            class FinalOutput(BaseModel):
                project_status: str
                completion_percentage: str
                final_assessment: Assessment
                deployment_readiness: DeploymentReadiness
                recommendations: TypeList[str]
            
            response, usage = self.remote_client.chat(
                messages=messages,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": FinalOutput,
                }
            )
        else:
            response, usage = self.remote_client.chat(messages=messages)
        
        # Parse final assessment
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient, GeminiClient)):
            try:
                assessment_data = json.loads(response[0])
            except:
                assessment_data = _extract_json(response[0])
        else:
            assessment_data = _extract_json(response[0])
        
        print(f"🎯 Final Assessment:")
        print(f"  Status: {assessment_data.get('project_status', 'unknown')}")
        print(f"  Completion: {assessment_data.get('completion_percentage', 'unknown')}")
        print(f"  Quality Score: {assessment_data.get('final_assessment', {}).get('quality_score', 'N/A')}")
        
        if self.callback:
            callback_message = {
                "role": "assistant",
                "content": f"Final Assessment Complete:\n\nStatus: {assessment_data.get('project_status', 'unknown')}\nCompletion: {assessment_data.get('completion_percentage', 'unknown')}\nQuality Score: {assessment_data.get('final_assessment', {}).get('quality_score', 'N/A')}"
            }
            self.callback("supervisor", callback_message, is_final=True)
        
        return {
            "assessment": assessment_data,
            "usage": usage,
        }
    
    def _format_workspace_state(self, state: Dict[str, Any]) -> str:
        """Format workspace state for inclusion in prompts."""
        if not state.get("files"):
            return "Empty workspace"
        
        formatted = f"Files ({len(state['files'])}):\n"
        for filepath in sorted(state["files"].keys()):
            lines = len(state["files"][filepath].splitlines())
            formatted += f"  - {filepath} ({lines} lines)\n"
        
        stats = state.get("statistics", {})
        formatted += f"\nStatistics:\n"
        formatted += f"  - Total files: {stats.get('total_files', 0)}\n"
        formatted += f"  - Total lines: {stats.get('total_lines', 0)}\n"
        
        return formatted
    
    def _format_completed_steps(self) -> str:
        """Format completed steps summary with test results."""
        if not self.completed_steps:
            return "No steps completed yet."
        
        summary = "Completed steps:\n"
        for step in self.completed_steps:
            status = "✅" if step["success"] else "❌"
            summary += f"  {status} Step {step['step_number']}: {step['step_title']}\n"
            
            # Include test results if available
            if step.get("test_results"):
                test_results = step["test_results"]
                if isinstance(test_results, dict):
                    if test_results.get("summary"):
                        summary += f"    🧪 Test Results: {test_results['summary']}\n"
                    if test_results.get("passed_tests"):
                        summary += f"    ✅ Passed: {test_results['passed_tests']}\n"
                    if test_results.get("failed_tests"):
                        summary += f"    ❌ Failed: {test_results['failed_tests']}\n"
                    if test_results.get("details"):
                        # Truncate details if too long
                        details = str(test_results['details'])
                        if len(details) > 200:
                            details = details[:200] + "..."
                        summary += f"    📋 Details: {details}\n"
            summary += "\n"
        
        return summary
    
    def _save_session_log(self, session_log: Dict[str, Any], logging_id: Optional[str]) -> str:
        """Save the development session log."""
        if logging_id:
            log_filename = f"{logging_id}_dev_session.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = "".join(c for c in session_log["task"][:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_task = safe_task.replace(' ', '_')
            log_filename = f"{timestamp}_{safe_task}_dev_session.json"
        
        log_path = os.path.join(self.log_dir, log_filename)
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(session_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save session log: {e}")
        
        return log_path 