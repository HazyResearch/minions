import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import difflib


class WorkspaceManager:
    """Manages a development workspace for DevMinion operations."""
    
    def __init__(self, workspace_dir: str, backup_enabled: bool = True):
        """
        Initialize workspace manager.
        
        Args:
            workspace_dir: Path to the workspace directory
            backup_enabled: Whether to create backups before modifications
        """
        self.workspace_dir = Path(workspace_dir)
        self.backup_enabled = backup_enabled
        self.logger = logging.getLogger(__name__)
        
        # Create workspace directory if it doesn't exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.docs_dir = self.workspace_dir / "docs"
        self.tests_dir = self.workspace_dir / "tests"
        self.backups_dir = self.workspace_dir / ".backups"
        
        for dir_path in [self.docs_dir, self.tests_dir, self.backups_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current state of the workspace.
        
        Returns:
            Dictionary containing file structure and contents
        """
        state = {
            "files": {},
            "structure": [],
            "statistics": {
                "total_files": 0,
                "total_lines": 0,
                "file_types": {}
            }
        }
        
        for file_path in self.workspace_dir.rglob("*"):
            if file_path.is_file() and not self._is_ignored_file(file_path):
                relative_path = file_path.relative_to(self.workspace_dir)
                state["structure"].append(str(relative_path))
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        state["files"][str(relative_path)] = content
                        
                        # Update statistics
                        state["statistics"]["total_files"] += 1
                        state["statistics"]["total_lines"] += len(content.splitlines())
                        
                        file_ext = file_path.suffix.lower()
                        if file_ext:
                            state["statistics"]["file_types"][file_ext] = \
                                state["statistics"]["file_types"].get(file_ext, 0) + 1
                        
                except (UnicodeDecodeError, PermissionError) as e:
                    self.logger.warning(f"Could not read file {relative_path}: {e}")
                    state["files"][str(relative_path)] = f"<Binary file or read error: {e}>"
        
        return state
    
    def get_file_contents(self, file_path: str) -> Optional[str]:
        """
        Get contents of a specific file in the workspace.
        
        Args:
            file_path: Path to the file relative to workspace directory
            
        Returns:
            File contents as string, or None if file doesn't exist or can't be read
        """
        full_path = self.workspace_dir / file_path
        
        # Check if file exists
        if not full_path.exists():
            self.logger.warning(f"File does not exist: {file_path}")
            return None
        
        # Check if path is actually a file
        if not full_path.is_file():
            self.logger.warning(f"Path is not a file: {file_path}")
            return None
        
        # Check if file should be ignored
        if self._is_ignored_file(full_path):
            self.logger.warning(f"File is ignored: {file_path}")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logger.info(f"Successfully read file: {file_path}")
                return content
        except UnicodeDecodeError as e:
            self.logger.warning(f"Could not decode file {file_path}: {e}")
            return None
        except PermissionError as e:
            self.logger.warning(f"Permission denied reading file {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error reading file {file_path}: {e}")
            return None
    
    def create_backup(self, label: str = None) -> str:
        """
        Create a backup of the current workspace state.
        
        Args:
            label: Optional label for the backup
            
        Returns:
            Path to the created backup
        """
        if not self.backup_enabled:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        if label:
            backup_name += f"_{label}"
        
        backup_path = self.backups_dir / backup_name
        
        # Create the backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy entire workspace (excluding backups directory)
        items_copied = 0
        for item in self.workspace_dir.iterdir():
            if item.name != ".backups":
                try:
                    if item.is_file():
                        shutil.copy2(item, backup_path / item.name)
                        items_copied += 1
                    elif item.is_dir():
                        shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)
                        items_copied += 1
                except Exception as e:
                    self.logger.warning(f"Failed to backup {item}: {e}")
        
        self.logger.info(f"Created backup: {backup_path} ({items_copied} items)")
        return str(backup_path)
    
    def apply_file_changes(self, files: Dict[str, str], sub_dir: str = None) -> Dict[str, str]:
        """
        Apply file changes to the workspace.
        
        Args:
            files: Dictionary mapping file paths to their content
            
        Returns:
            Dictionary of changes made (file -> change type)
        """
        changes = {}
        
        for file_path, content in files.items():
            if sub_dir:
                full_path = self.workspace_dir / sub_dir / file_path
            else:
                full_path = self.workspace_dir / file_path
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine if this is a new file or modification
            if full_path.exists():
                changes[file_path] = "modified"
            else:
                changes[file_path] = "created"
            
            # Write the file
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Applied changes to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to write file {file_path}: {e}")
                changes[file_path] = f"error: {e}"
        
        return changes
    
    def run_tests(self, test_commands: List[str] = None) -> Dict[str, Any]:
        """
        Run tests in the workspace.
        
        Args:
            test_commands: List of test commands to run
            
        Returns:
            Dictionary containing test results
        """
        if test_commands is None:
            # Auto-detect common test commands
            test_commands = self._detect_test_commands()
        
        results = {
            "commands_run": [],
            "results": {},
            "overall_success": True,
            "summary": ""
        }
        
        for command in test_commands:
            self.logger.info(f"Running test command: {command}")
            
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                command_result = {
                    "command": command,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.returncode == 0
                }
                
                results["commands_run"].append(command)
                results["results"][command] = command_result
                
                if result.returncode != 0:
                    results["overall_success"] = False
                
            except subprocess.TimeoutExpired:
                command_result = {
                    "command": command,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": "Command timed out after 5 minutes",
                    "success": False
                }
                results["results"][command] = command_result
                results["overall_success"] = False
            
            except Exception as e:
                command_result = {
                    "command": command,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "success": False
                }
                results["results"][command] = command_result
                results["overall_success"] = False
        
        # Generate summary
        total_commands = len(results["commands_run"])
        successful_commands = sum(1 for r in results["results"].values() if r["success"])
        results["summary"] = f"{successful_commands}/{total_commands} test commands passed"
        
        return results
    
    def generate_diff(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate diffs between workspace states.
        
        Args:
            before_state: Previous workspace state
            after_state: Current workspace state
            
        Returns:
            Dictionary mapping file paths to their diffs
        """
        diffs = {}
        
        # Find all files that were added, modified, or deleted
        before_files = set(before_state.get("files", {}).keys())
        after_files = set(after_state.get("files", {}).keys())
        
        # New files
        for file_path in after_files - before_files:
            diffs[file_path] = f"NEW FILE:\n{after_state['files'][file_path]}"
        
        # Deleted files
        for file_path in before_files - after_files:
            diffs[file_path] = f"DELETED FILE:\n{before_state['files'][file_path]}"
        
        # Modified files
        for file_path in before_files & after_files:
            before_content = before_state["files"][file_path]
            after_content = after_state["files"][file_path]
            
            if before_content != after_content:
                diff_lines = list(difflib.unified_diff(
                    before_content.splitlines(keepends=True),
                    after_content.splitlines(keepends=True),
                    fromfile=f"before/{file_path}",
                    tofile=f"after/{file_path}"
                ))
                diffs[file_path] = "".join(diff_lines)
        
        return diffs
    
    def create_documentation_file(self, step_number: int, content: str) -> str:
        """
        Create a documentation file for a development step.
        
        Args:
            step_number: The step number
            content: The markdown content
            
        Returns:
            Path to the created documentation file
        """
        doc_filename = f"step_{step_number:02d}_implementation.md"
        doc_path = self.docs_dir / doc_filename
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Created documentation: {doc_path}")
        return str(doc_path.relative_to(self.workspace_dir))
    
    def get_project_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the project in the workspace.
        
        Returns:
            Dictionary containing project summary information
        """
        state = self.get_current_state()
        
        summary = {
            "file_count": state["statistics"]["total_files"],
            "line_count": state["statistics"]["total_lines"],
            "file_types": state["statistics"]["file_types"],
            "main_files": [],
            "documentation_files": [],
            "test_files": [],
            "config_files": [],
            "files": state["files"]
        }
        
        # Categorize files
        for file_path in state["structure"]:
            file_path_lower = file_path.lower()
            
            if any(name in file_path_lower for name in ["main.py", "app.py", "index.py", "server.py"]):
                summary["main_files"].append(file_path)
            elif file_path.endswith(".md") or "docs/" in file_path_lower:
                summary["documentation_files"].append(file_path)
            elif "test" in file_path_lower or file_path.endswith("_test.py"):
                summary["test_files"].append(file_path)
            elif any(name in file_path_lower for name in ["requirements.txt", "package.json", "setup.py", "pyproject.toml"]):
                summary["config_files"].append(file_path)
        
        return summary
    
    def _detect_test_commands(self) -> List[str]:
        """
        Auto-detect appropriate test commands based on project structure.
        
        Returns:
            List of test commands to try
        """
        commands = []
        state = self.get_current_state()
        
        # Check for Python projects
        if any(f.endswith(".py") for f in state["structure"]):
            # Check for specific test frameworks
            if "pytest.ini" in state["files"] or any("pytest" in f for f in state["structure"]):
                commands.append("pytest -v")
            elif any("unittest" in content for content in state["files"].values()):
                commands.append("python -m unittest discover -v")
            else:
                commands.append("python -m pytest -v")  # Default Python testing
        
        # Check for Node.js projects
        if "package.json" in state["files"]:
            commands.append("npm test")
        
        # Check for other common test setups
        if "Makefile" in state["files"] and "test" in state["files"]["Makefile"]:
            commands.append("make test")
        
        # Fallback - just try to run main files to see if they work (only works if it's not an app since not all have --help)
        # FIX
        for file_path in state["structure"]:
            if file_path.lower() in ["main.py", "app.py"]:
                #commands.append(f"python {file_path} --help")  # Basic syntax check
                continue
        
        return commands if commands else ["echo 'No tests detected'"]
    
    def _is_ignored_file(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored (e.g., temporary files, cache).
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be ignored
        """
        ignored_patterns = [
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".DS_Store",
            "*.pyc",
            "*.pyo",
            ".backups"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignored_patterns)
    
    def cleanup(self):
        """Clean up temporary files and old backups."""
        # Remove old backups (keep only last 10)
        if self.backups_dir.exists():
            backups = sorted(self.backups_dir.iterdir(), key=lambda x: x.stat().st_mtime)
            for old_backup in backups[:-10]:  # Keep last 10 backups
                if old_backup.is_dir():
                    shutil.rmtree(old_backup)
                else:
                    old_backup.unlink() 