import json
import re
import os
import subprocess
import time

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from minions.minion import Minion
from minions.clients import OpenAIClient, TogetherClient

from minions.prompts.minion_cua import (
    WORKER_CUA_SYSTEM_PROMPT,
    SUPERVISOR_CUA_REVIEW_PROMPT,
    SUPERVISOR_CUA_ANNOUNCEMENT_PROMPT,
    SUPERVISOR_CUA_SUMMARY_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_CUA_INITIAL_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL
)

from minions.usage import Usage


# --- Constants for Safety Rules ---

# Applications allowed to receive keystrokes (restrictive for safety)
KEYSTROKE_ALLOWED_APPS = [
    "Calculator", 
    "TextEdit", 
    "Notes", 
    "Terminal", 
    "Safari", 
    "Chrome", 
    "Firefox",
    "Mail",
    "Messages",
    "Slack",
    "Microsoft Word",
    "Microsoft Excel",
    "Microsoft PowerPoint",
    "Visual Studio Code",
    "Finder",
    "Google Chrome" # Added for Gmail login
]

# Special app-specific character sets for keystroke safety
APP_SPECIFIC_ALLOWED_KEYS = {
    "Calculator": set("0123456789+-*/=().c"),  # Calculator keys
    # For other apps, we'll use a general safe character set
}

# Default set of allowed keys for general text entry
DEFAULT_ALLOWED_KEYS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-_+=@#$%^&*[]{}|\\/<>`~\n\t")

# Safe keyboard shortcuts (modifiers + key)
SAFE_KEY_COMBOS = {
    "command+n",     # New Document/Window
    "command+t",     # New tab
    "command+w",     # Close tab/window
    "command+r",     # Refresh
    "command+a",     # Select all
    "command+c",     # Copy
    "command+v",     # Paste
    "command+x",     # Cut
    "command+z",     # Undo
    "command+s",     # Save
    "command+f",     # Find
    "command+p",     # Print
    "command+shift+t", # Reopen closed tab
    "command+shift+n", # New Incognito/Private window
    "command+option+i", # Developer tools
    "command+plus",  # Zoom in
    "command+minus", # Zoom out
    "command+0",     # Reset zoom
    "command+left",  # Navigate back
    "command+right", # Navigate forward
    "command+shift+tab", # Previous tab
    "command+tab",   # Next tab
}

# Unsafe operations we'll explicitly block
UNSAFE_KEY_COMBOS = {
    "command+space",    # Spotlight (system-wide)
    "command+option+esc", # Force Quit
    "command+shift+q",  # Log out
    "command+shift+3",  # Screenshot
    "command+shift+4",  # Screenshot selection
    "command+q",        # Quit app
    "command+h",        # Hide app
    "command+m",        # Minimize app
    "command+option+m", # Minimize all windows
    "fn+f11",           # Show desktop
    "command+option+power", # Sleep
    "command+option+control+power", # Shutdown
}

# Safe websites allow list (domains only, for security)
SAFE_WEBSITE_DOMAINS = {
    "google.com",
    "gmail.com",
    "mail.google.com", # Added for Gmail login
    "youtube.com",
    "github.com",
    "apple.com",
    "wikipedia.org",
    "microsoft.com",
    "office.com",
    "linkedin.com",
    "amazon.com",
    "netflix.com",
    "nytimes.com",
    "cnn.com",
    "bbc.com",
    "weather.com",
    "yahoo.com",
    "dropbox.com",
    "drive.google.com",
    "docs.google.com",
    "sheets.google.com",
    "calendar.google.com",
    "maps.google.com",
    "twitter.com",
    "instagram.com",
    "facebook.com",
    "reddit.com",
    "accounts.google.com", # Added for Gmail login
}
# --- End Constants ---


class MinionCUA(Minion):
    """
    Minion with Computer User Automation (CUA) capabilities for macOS.

    WARNING: This class allows the language model to execute actions directly
    on the user's computer (opening apps, typing, clicking, running scripts)
    via AppleScript. This functionality is EXPERIMENTAL and carries SIGNIFICANT
    SECURITY RISKS. Enabling this allows the LLM to potentially interact with
    any application, access sensitive information, or perform unintended actions.
    Use with extreme caution and only in controlled environments.

    NOTE: UI Automation via AppleScript is often fragile and may break due to
    OS updates, application updates, or UI changes. Actions may fail unexpectedly.
    """
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
    ):
        """Initialize the MinionCUA with local and remote LLM clients."""
        super().__init__(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=max_rounds,
            callback=callback,
            log_dir=log_dir,
        )
        if not self.local_client or not self.remote_client:
             print("MinionCUA requires both local_client and remote_client.")
             raise ValueError("MinionCUA requires both local_client and remote_client.")
        print("MinionCUA initialized with expanded automation capabilities.")
        
        # Keep track of action history for summaries
        self.action_history = []
        self.credentials_store = {} 

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds: Optional[int] = None,
        doc_metadata=None,
        logging_id=None,
        is_privacy=False,
        images=None,
        use_bm25=False,
    ) -> Dict[str, Any]:
        """Run the MinionCUA protocol, returning a result dictionary."""

        # --- Initialization ---
        print(f"\n========== MINION-CUA TASK STARTED ==========")
        print(f"Task: {task}")
        current_max_rounds = max_rounds if max_rounds is not None else self.max_rounds
        print(f"Max rounds: {current_max_rounds}")
        context_str = "\n\n".join(context)
        print(f"Context length: {len(context_str)} characters")

        # Reset action history for this new task
        self.action_history = []   

        conversation_log = { 
            "task": task, 
            "context": context_str, 
            "conversation": [], 
            "automation_actions": [], 
            "final_answer": None, 
            "usage": {"remote": {}, "local": {}} 
        }
        remote_usage = Usage()
        local_usage = Usage()
        final_answer = "Processing failed to complete." # Default fail answer

        # --- Initial Worker Prompt Setup ---
        try:
            worker_system_prompt = WORKER_CUA_SYSTEM_PROMPT.format(context=context_str, task=task)
        except KeyError as e:
            print(f"Formatting WORKER_CUA_SYSTEM_PROMPT failed. Missing key: {e}")
            worker_system_prompt = f"Task: {task}\nContext: {context_str}\n(Error: Prompt template failed)"

        worker_messages = [{"role": "system", "content": worker_system_prompt}]

        # --- Extract credentials from task/context if available ---
        self.extract_credentials(task, context_str)  

        if images: 
            print(f"Received {len(images)} images.")

        # --- Initial Supervisor Call ---
        # Use our special CUA-specific initial prompt
        supervisor_initial_prompt = SUPERVISOR_CUA_INITIAL_PROMPT.format(
            task=task, 
            max_rounds=current_max_rounds
        )
        
        # Add an explicit system message to inform the model about automation capabilities
        supervisor_messages = [
            {
                "role": "system", 
                "content": """IMPORTANT: You are supervising a system with REAL physical automation capabilities.
                The worker model can directly control the user's Mac computer to open apps, type text, click elements,
                use keyboard shortcuts, and open URLs. These are not instructions for the user - they are actions
                that will be physically executed on the computer."""
            },
            {"role": "user", "content": supervisor_initial_prompt}
        ]
        
        conversation_log["conversation"].append(
            {"user": "remote", "prompt": supervisor_initial_prompt, "output": None}
        )

        if self.callback: 
            self.callback("supervisor", None, is_final=False)
            
        try:
            response_format_arg = {"response_format": {"type": "json_object"}} if isinstance(self.remote_client, (OpenAIClient, TogetherClient)) else {}
            supervisor_response, usage = self.remote_client.chat(messages=supervisor_messages, **response_format_arg)
            remote_usage += usage
            supervisor_reply_content = supervisor_response[0]
            supervisor_messages.append({"role": "assistant", "content": supervisor_reply_content})
            conversation_log["conversation"][-1]["output"] = supervisor_reply_content
            if self.callback: 
                self.callback("supervisor", supervisor_messages[-1])

            supervisor_json = self._extract_json(supervisor_reply_content, "supervisor initial response")
            first_worker_prompt = supervisor_json.get("message") if supervisor_json and isinstance(supervisor_json, dict) else None
            if not first_worker_prompt:
                first_worker_prompt = "Based on the task and context, what action, if any, should be taken?"

        except Exception as e:
            final_answer = f"An error occurred during the initial setup: {e}"
            if self.callback: 
                self.callback("supervisor", {"role": "assistant", "content": final_answer}, is_final=True)
            # Return error dictionary immediately
            return {
                "final_answer": final_answer, 
                "local_usage": local_usage, 
                "remote_usage": remote_usage, 
                "conversation_log": conversation_log
            }

        worker_messages.append({"role": "user", "content": first_worker_prompt})
        conversation_log["conversation"].append({"user": "local", "prompt": first_worker_prompt, "output": None})

        # --- Main Interaction Loop ---
        try: # Wrap main loop in try/except
            for round_num in range(current_max_rounds):
                action_succeeded = True # Assume success unless action fails

                # 1. Get Worker Response
                if self.callback: 
                    self.callback("worker", None, is_final=False)
                    
                worker_response, usage, _ = self.local_client.chat(messages=worker_messages)
                local_usage += usage
                worker_reply_content = worker_response[0]
                worker_messages.append({"role": "assistant", "content": worker_reply_content})
                conversation_log["conversation"][-1]["output"] = worker_reply_content
                if self.callback: 
                    self.callback("worker", worker_messages[-1])

                # 2. Parse Worker Response
                worker_action_json = self._extract_json(worker_reply_content, f"worker response round {round_num+1}")

                # 3. Action Handling
                action_executed_this_round = False # Track if an action was attempted
                action_type = worker_action_json.get("action") if worker_action_json else None

                # Handle all action types, not just the original limited set
                if action_type in ["open_app", "type_keystrokes", "click_element", "key_combo", "open_url", "none"]:
                    action_executed_this_round = True # Mark that we are attempting an action

                    # 3a. Internal Safety Check
                    if self.is_action_safe(worker_action_json):

                        # 3b. Supervisor Review
                        try:
                            response_dump = json.dumps(worker_action_json)
                            review_prompt = SUPERVISOR_CUA_REVIEW_PROMPT.format(response=response_dump)
                        except (KeyError, TypeError) as prompt_err:
                            final_answer = f"Internal error processing action proposal: {prompt_err}"
                            action_succeeded = False
                            break # Exit loop on internal error

                        supervisor_messages.append({"role": "user", "content": review_prompt})
                        conversation_log["conversation"].append({"user": "remote", "prompt": review_prompt, "output": None})
                        if self.callback: 
                            self.callback("supervisor", None, is_final=False)
                        # Get Review
                        try:
                            response_format_arg = {"response_format": {"type": "json_object"}} if isinstance(self.remote_client, (OpenAIClient, TogetherClient)) else {}
                            review_response, usage = self.remote_client.chat(messages=supervisor_messages, **response_format_arg)
                            remote_usage += usage
                            review_reply_content = review_response[0]
                            supervisor_messages.append({"role": "assistant", "content": review_reply_content})
                            conversation_log["conversation"][-1]["output"] = review_reply_content
                            if self.callback: 
                                self.callback("supervisor", supervisor_messages[-1])
                            review_json = self._extract_json(review_reply_content, "supervisor review response")
                        except Exception as chat_err:
                            if self.callback: 
                                self.callback("supervisor", {"role": "assistant", "content": f"Error getting review: {chat_err}"}, is_final=True)
                            review_json = None # Ensure review_json is None
                            final_answer = f"Error getting supervisor review: {chat_err}"
                            action_succeeded = False
                            break # Exit loop if review fails

                        # 3c. Parse Review & Process Approval
                        if review_json and review_json.get("is_safe") is True:
                            approved_action = review_json.get("approved_action")

                            if approved_action and isinstance(approved_action, dict):
                                # 3d. Announcement
                                action_details_str = json.dumps(approved_action)
                                try:
                                    announce_prompt = SUPERVISOR_CUA_ANNOUNCEMENT_PROMPT.format(action_details=action_details_str)
                                    # Use temporary message list for announcement to avoid polluting main history with prompt
                                    announce_messages = supervisor_messages + [{"role": "user", "content": announce_prompt}]
                                    announce_response, usage = self.remote_client.chat(messages=announce_messages)
                                    remote_usage += usage
                                    announce_reply_content = announce_response[0].strip()
                                    if self.callback: 
                                        self.callback("supervisor", {"role": "assistant", "content": announce_reply_content})
                                except Exception as announce_err:
                                    if self.callback: 
                                        self.callback("supervisor", {"role": "assistant", "content": f"(Error generating announcement: {announce_err}) Proceeding..."})

                                # --- 3e. Execute Action ---
                                # Include credential lookup for login actions
                                if action_type == "login_to_gmail":
                                    # Inject credentials for Gmail login if available
                                    approved_action = self.inject_credentials(approved_action, "gmail")

                                success, action_result_msg = self._execute_action(approved_action)

                                # Log action attempt
                                log_entry = {**self.sanitize_credentials_for_display(approved_action), "round": round_num + 1, "result": action_result_msg}
                                conversation_log["automation_actions"].append(log_entry)

                                if success:
                                    # Inform Worker only on Success
                                    result_message_to_worker = f"Action '{approved_action['action']}' executed successfully. Result: {action_result_msg}. Continue task if needed."
                                    worker_messages.append({"role": "user", "content": result_message_to_worker})
                                    conversation_log["conversation"].append({"user": "local", "prompt": result_message_to_worker, "output": None})
                                    # Let loop continue
                                else:
                                    # Action Failed! Stop the loop and report error.
                                    final_answer = action_result_msg
                                    if self.callback: 
                                        self.callback("supervisor", {"role": "assistant", "content": final_answer}, is_final=True)
                                    action_succeeded = False
                                    break 
                            else: # approved_action not a dict
                                rejection_reason = review_json.get("reasoning", "Approval format incorrect or invalid.") if review_json else "Approval format incorrect."
                                if self.callback: 
                                    self.callback("supervisor", {"role": "assistant", "content": f"Action approved but structure invalid: {rejection_reason}"})
                        else:
                            # Supervisor rejected or review parsing failed
                            rejection_reason = review_json.get("reasoning", "Action deemed unsafe or review failed.") if review_json else "Could not parse supervisor review."
                            if self.callback: 
                                self.callback("supervisor", {"role": "assistant", "content": f"Action rejected: {rejection_reason}"})

                    else: # Internal safety check failed
                        if self.callback: 
                            self.callback("supervisor", {"role": "assistant", "content": "Proposed action is not allowed for security reasons."})

                elif action_type and action_type != "none":
                    # Worker proposed an unknown or unsupported action type
                    if self.callback: 
                        self.callback("supervisor", {"role": "assistant", "content": f"Sorry, I don't support the action '{action_type}'. Please try a different approach."})

                # --- If loop broken by failed action, skip supervisor synthesis ---
                if not action_succeeded:
                    break # Ensure we exit the loop

                # --- 4. Supervisor Synthesis/Next Question (Only if no action failure) ---
                last_worker_output = worker_messages[-1]["content"]
                if round_num == current_max_rounds - 1:
                    supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(response=last_worker_output)
                    is_final_supervisor_call = True
                else:
                    # Intermediate round synthesis
                    cot_prompt = REMOTE_SYNTHESIS_COT.format(response=last_worker_output)
                    supervisor_messages.append({"role": "user", "content": cot_prompt})
                    conversation_log["conversation"].append({"user": "remote", "prompt": cot_prompt, "output": None})
                    try:
                        step_by_step_response, usage = self.remote_client.chat(supervisor_messages)
                        remote_usage += usage
                        step_by_step_content = step_by_step_response[0]
                        supervisor_messages.append({"role": "assistant", "content": step_by_step_content})
                        conversation_log["conversation"][-1]["output"] = step_by_step_content
                    except Exception as e:
                        step_by_step_content = last_worker_output # Fallback
                    supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(response=step_by_step_content)
                    is_final_supervisor_call = False

                supervisor_messages.append({"role": "user", "content": supervisor_prompt})
                conversation_log["conversation"].append({"user": "remote", "prompt": supervisor_prompt, "output": None})
                if self.callback: 
                    self.callback("supervisor", None, is_final=False)

                # Get Supervisor Decision
                try:
                    response_format_arg = {"response_format": {"type": "json_object"}} if isinstance(self.remote_client, (OpenAIClient, TogetherClient)) else {}
                    supervisor_decision_response, usage = self.remote_client.chat(messages=supervisor_messages, **response_format_arg)
                    remote_usage += usage
                    supervisor_decision_content = supervisor_decision_response[0]
                    supervisor_messages.append({"role": "assistant", "content": supervisor_decision_content})
                    conversation_log["conversation"][-1]["output"] = supervisor_decision_content
                    if self.callback: 
                        self.callback("supervisor", supervisor_messages[-1]) # Show the decision JSON
                except Exception as chat_err:
                    final_answer = f"Error getting supervisor decision: {chat_err}"
                    action_succeeded = False
                    break # Exit loop if decision fails

                # Parse decision
                try:
                    supervisor_decision_json = self._extract_json(supervisor_decision_content, f"supervisor decision round {round_num+1}")
                    
                    if supervisor_decision_json:
                        decision = supervisor_decision_json.get("decision", "request_additional_info")
                        message = supervisor_decision_json.get("message", "") # Question for worker
                        answer = supervisor_decision_json.get("answer", "")   # Final answer
                        
                        # Ensure we have valid strings for message and answer
                        message = str(message) if message is not None else ""
                        answer = str(answer) if answer is not None else ""
                    else:
                        # Try to extract decision using regex patterns
                        decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', supervisor_decision_content)
                        message_match = re.search(r'"message"\s*:\s*"([^"]+)"', supervisor_decision_content)
                        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', supervisor_decision_content)
                        
                        decision = decision_match.group(1) if decision_match else "request_additional_info"
                        message = message_match.group(1) if message_match else ""
                        answer = answer_match.group(1) if answer_match else ""
                except Exception as parsing_error:
                    decision = "request_additional_info"
                    message = ""
                    answer = "Error processing supervisor's response"

                # Process the decision
                if is_final_supervisor_call or decision == "provide_final_answer":
                    # Save the intermediate answer but don't break the loop yet
                    # We'll generate a better summary at the end
                    intermediate_answer = answer if answer else (message or "Processing complete. No specific answer provided.")
                    break # Exit the interaction loop
                elif message:
                    worker_messages.append({"role": "user", "content": message})
                    conversation_log["conversation"].append({"user": "local", "prompt": message, "output": None})
                    # Let loop continue
                else:
                    final_answer = "Sorry, I got stuck. Let's try again."
                    action_succeeded = False
                    break
        except Exception as loop_error:
            # Catch errors during the main loop execution (e.g., LLM client errors)
            final_answer = f"An error occurred during processing: {loop_error}"
            # Loop terminates due to exception

        # --- End of Loop / Generate Final Summary ---
        # If we have a successful execution with action history, generate a proper summary
        if action_succeeded and self.action_history:
            try:
                action_history_str = json.dumps(self.action_history, indent=2)
                summary_prompt = SUPERVISOR_CUA_SUMMARY_PROMPT.format(
                    original_task=task,
                    action_history=action_history_str
                )
                
                summary_messages = [{"role": "user", "content": summary_prompt}]
                
                if self.callback: 
                    self.callback("supervisor", None, is_final=False)
                    
                summary_response, usage = self.remote_client.chat(messages=summary_messages)
                remote_usage += usage
                final_answer = summary_response[0]
                
                
                if self.callback: 
                    self.callback("supervisor", {"role": "assistant", "content": final_answer}, is_final=True)
            except Exception as summary_err:
                # Fall back to the intermediate answer if we have one
                if 'intermediate_answer' in locals():
                    final_answer = intermediate_answer
                # Otherwise use what we already have in final_answer
                    

        conversation_log["final_answer"] = final_answer
        if logging_id:
            log_filename = f"{logging_id}_minion_cua.json"
        else:
            # Fallback to timestamp + short safe filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}_cua.json"

        os.makedirs(self.log_dir, exist_ok=True) # <--- ADD THIS LINE
        log_path = os.path.join(self.log_dir, log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        return {
            "final_answer": final_answer,
            "local_usage": local_usage,
            "remote_usage": remote_usage,
            "conversation_log": conversation_log,
            "action_history": self.action_history,
            "log_file": log_path, 
        }
    
    def _ensure_new_textedit_document(self) -> Tuple[bool, str]:
        """
        Robust method to ensure a new TextEdit document is created.
        
        :return: Tuple of (success, message)
        """
        
        create_doc_script = '''
        tell application "System Events"
            if not (exists process "TextEdit") then
                tell application "TextEdit" to activate
                delay 1
            end if
            
            tell process "TextEdit"
                try
                    -- Try menu bar method
                    click menu item "New" of menu "File" of menu bar item "File" of menu bar 1
                    delay 0.5
                    return "menu_success"
                on error
                    try
                        -- Try keyboard shortcut
                        keystroke "n" using {command down}
                        delay 0.5
                        return "shortcut_success"
                    on error
                        return "both_failed"
                    end try
                end try
            end tell
        end tell
        '''
        
        try:
            result = subprocess.run(
                ['osascript', '-e', create_doc_script], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            output = result.stdout.strip()
            stderr = result.stderr.strip()


            if output in ["menu_success", "shortcut_success"]:
                return True, f"New document created ({output})"
            else:
                return False, "Could not create new document"

        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _extract_json(self, text: str, description: str = "LLM response") -> Optional[Dict[str, Any]]:
        """Clean common LLM JSON issues and parse."""
        if not text:
            return None
        cleaned_text = text.strip()

        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()

        # Try to extract JSON object using regex pattern
        start = cleaned_text.find('{')
        end = cleaned_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            cleaned_text = cleaned_text[start : end + 1]
        else:
            return None

        try:
            # First try standard JSON parsing
            parsed_json = json.loads(cleaned_text)
            if isinstance(parsed_json, dict):
                return parsed_json
            else:
                return None
        except json.JSONDecodeError as e:
            try:
                # Remove newlines and normalize spaces
                simplified_text = re.sub(r'\s+', ' ', cleaned_text)
                parsed_json = json.loads(simplified_text)
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
            except json.JSONDecodeError:
                # Still failed, give up
                return None
        
    def is_action_safe(self, action_json: Optional[Dict[str, Any]]) -> bool:
        """
        Enhanced safety validation with more flexible UI interaction support.
        """
        if not action_json or not isinstance(action_json, dict):
            return False

        try:
            action = action_json.get("action")
            app_name = action_json.get("app_name")

            # Expanded list of safe action types
            SAFE_ACTION_TYPES = [
                "open_app", 
                "type_keystrokes", 
                "click_element", 
                "key_combo", 
                "open_url", 
                "menu_click", 
                "login_to_gmail",  
                "none"
            ]

            # Check if action type is valid
            if action not in SAFE_ACTION_TYPES:
                return False
            
            # Handle 'none' action
            if action == "none":
                explanation = action_json.get("explanation")
                return explanation and isinstance(explanation, str)

            # Validate app name for actions that require it
            if action in ["open_app", "type_keystrokes", "click_element", "key_combo", "menu_interaction"]:
                if not app_name or not isinstance(app_name, str):
                    return False

            # Specific validation for different action types
            if action == "open_app":
                return True

            elif action == "type_keystrokes":
                # Support both 'keys' and 'text' parameters
                keys_to_type = action_json.get("keys") or action_json.get("text")
                
                # Validate keys exist and are a string
                if not keys_to_type or not isinstance(keys_to_type, str):
                    return False

                # More lenient character validation
                if len(keys_to_type) > 500:
                    return False
                return True

            elif action == "click_element":
                # Support clicking by description or coordinates
                element_desc = action_json.get("element_desc")
                coordinates = action_json.get("coordinates")

                # Require either element description or coordinates
                if not element_desc and not coordinates:
                    return False

                # If coordinates provided, validate them
                if coordinates:
                    try:
                        x, y = float(coordinates[0]), float(coordinates[1])
                    except (TypeError, ValueError, IndexError):
                        return False
                return True

            elif action == "key_combo":
                # Validate combo is a list of modifiers and keys
                combo = action_json.get("combo")
                
                if not combo or not isinstance(combo, list):
                    return False

                # Allowed modifiers and keys
                ALLOWED_MODIFIERS = {"command", "control", "option", "shift"}
                ALLOWED_KEYS = set("abcdefghijklmnopqrstuvwxyz0123456789")

                # Validate each part of the combo
                for key in combo:
                    if key.lower() not in ALLOWED_MODIFIERS and key.lower() not in ALLOWED_KEYS:
                        return False

                # Prevent only the most dangerous key combinations
                UNSAFE_COMBOS = {
                    "command+space",  # Spotlight
                    "command+option+esc",  # Force Quit
                    "command+shift+q"  # Log out
                }
                
                combo_str = "+".join(combo).lower()
                if combo_str in UNSAFE_COMBOS:
                    return False
                return True

            elif action == "open_url":
                url = action_json.get("url")
                
                if not url or not isinstance(url, str):
                    return False

                # Basic URL validation
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    
                    # Check protocol
                    if parsed_url.scheme not in ['http', 'https']:
                        return False
                    return True
                except Exception as e:
                    return False

            elif action == "menu_click":
                # Validate menu parameters
                menu_name = action_json.get("menu_name")
                menu_item = action_json.get("menu_item")
                
                if not menu_name or not isinstance(menu_name, str):
                    return False
                    
                if not menu_item or not isinstance(menu_item, str):
                    return False
                
                # Menu items shouldn't be too long (arbitrary safety check)
                if len(menu_name) > 50 or len(menu_item) > 50:
                    return False
                return True
                
            elif action == "login_to_gmail":
                # Validate login parameters
                browser = action_json.get("browser")
                
                # Browser is required
                if not browser or not isinstance(browser, str):
                    return False
                
                # Browser must be one of the supported browsers
                if browser.lower() not in ["safari", "chrome", "firefox", "google chrome"]:
                    return False
                return True
            return False
        except Exception as e:
            return False

    def open_app(self, app_name: str) -> tuple[bool, str]:
        """Open a macOS application using the 'open' command."""
        try:
            # Use the 'open' command
            result = subprocess.run(
                ['open', '-a', app_name], # Command: open -a "AppName"
                check=True,
                capture_output=True,
                text=True,
                timeout=15
            )
            return True, f"Successfully opened/activated {app_name}."

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else f"'open -a {app_name}' command failed with no stderr."
            if "Unable to find application" in stderr_output:
                error_msg = f"Error: Application '{app_name}' could not be found by the 'open' command."
            else:
                error_msg = f"Error: Failed to open {app_name} using 'open' command. Details: {stderr_output}"
            return False, error_msg

        except FileNotFoundError:
            return False, "Error: The 'open' command was not found on this system."

        except subprocess.TimeoutExpired:
            return False, f"Error: Timeout waiting for {app_name} to respond via 'open' command."

        except Exception as e:
            return False, f"Error: An unexpected error ({type(e).__name__}) occurred trying to open {app_name}."
        
    def click_element(self, app_name: str, element_desc: Optional[str] = None, 
                     coordinates: Optional[List[int]] = None) -> tuple[bool, str]:
        """Click on a UI element in an application by description or coordinates."""
        try:
            # Ensure the target app is frontmost first
            subprocess.run(['open', '-a', app_name], check=True, capture_output=True, text=True, timeout=10)
            time.sleep(0.5)  # Pause for activation

            # Select the appropriate script based on whether we're clicking by element or coordinates
            if element_desc:
                # Click by UI element description
                click_script = f'''
                    tell application "System Events"
                        if not (UI elements enabled) then
                            return "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                        end if
                        if not (exists process "{app_name}") then
                            return "Error: Process '{app_name}' does not exist. Cannot click element."
                        end if

                        try
                            tell process "{app_name}"
                                set frontmost to true
                                delay 0.3
                                
                                -- Try to find the element by description
                                set foundElement to false
                                
                                -- Try to find by name
                                if not foundElement then
                                    set theElements to UI elements whose name contains "{element_desc}"
                                    if (count of theElements) > 0 then
                                        click item 1 of theElements
                                        set foundElement to true
                                        return "Successfully clicked element with name: {element_desc}"
                                    end if
                                end if
                                
                                -- Try to find buttons
                                if not foundElement then
                                    set theButtons to buttons whose name contains "{element_desc}"
                                    if (count of theButtons) > 0 then
                                        click item 1 of theButtons
                                        set foundElement to true
                                        return "Successfully clicked button: {element_desc}"
                                    end if
                                end if
                                
                                -- Try to find text fields
                                if not foundElement then
                                    set theFields to text fields whose name contains "{element_desc}"
                                    if (count of theFields) > 0 then
                                        click item 1 of theFields
                                        set foundElement to true
                                        return "Successfully clicked text field: {element_desc}"
                                    end if
                                end if
                                
                                -- Check if we found and clicked the element
                                if not foundElement then
                                    return "Error: Could not find element containing description: {element_desc}"
                                end if
                            end tell
                        on error errMsg number errorNumber
                            return "Error during click: " & errMsg & " (" & errorNumber & ")"
                        end try
                    end tell
                '''
            elif coordinates:
                # Click by absolute coordinates
                x, y = coordinates
                click_script = f'''
                    tell application "System Events"
                        if not (UI elements enabled) then
                            return "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                        end if
                        if not (exists process "{app_name}") then
                            return "Error: Process '{app_name}' does not exist. Cannot perform click."
                        end if

                        try
                            tell process "{app_name}"
                                set frontmost to true
                                delay 0.3
                            end tell
                            
                            -- Click at the specified coordinates
                            tell application "{app_name}"
                                activate
                            end tell
                            delay 0.2
                            
                            -- Get screen size to validate coordinates
                            set screenWidth to do shell script "system_profiler SPDisplaysDataType | grep Resolution | awk '{{print $2}}' | head -1"
                            set screenHeight to do shell script "system_profiler SPDisplaysDataType | grep Resolution | awk '{{print $4}}' | head -1"
                            
                            -- Convert to numbers
                            set screenWidth to screenWidth as number
                            set screenHeight to screenHeight as number
                            
                            -- Validate coordinates are within screen bounds
                            if {x} < 0 or {x} > screenWidth or {y} < 0 or {y} > screenHeight then
                                return "Error: Coordinates ({x}, {y}) are outside screen bounds (" & screenWidth & "x" & screenHeight & ")"
                            end if
                            
                            -- Perform the click
                            click at {{{x}, {y}}}
                            return "Successfully clicked at coordinates: ({x}, {y})"
                        on error errMsg number errorNumber
                            return "Error during click: " & errMsg & " (" & errorNumber & ")"
                        end try
                    end tell
                '''
            else:
                # We should never reach here due to previous validation
                return False, "Error: Neither element description nor coordinates provided for click operation."

            # Execute the script
            result = subprocess.run(
                ['osascript', '-e', click_script],
                capture_output=True,
                text=True,
                timeout=20
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # Check for errors
            if stderr:
                return False, f"Error: osascript failed for click operation. Stderr: {stderr}"

            if stdout.startswith("Error:"):
                return False, stdout
            elif "Successfully clicked" in stdout:
                return True, stdout
            else:
                return False, f"Unexpected result from click operation: {stdout}"

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else f"'open -a {app_name}' command failed."
            return False, f"Error activating or clicking in {app_name}. Details: {stderr_output}"
        except FileNotFoundError:
            return False, "Error: Required command ('osascript' or 'open') not found."
        except subprocess.TimeoutExpired:
            return False, f"Error: Timeout waiting for {app_name} to respond to click operation."
        except Exception as e:
            return False, f"Error: An unexpected error ({type(e).__name__}) occurred trying to click in {app_name}."
    
    
    def _execute_action(self, action_json: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Execute the appropriate action based on the action type in the JSON.

        WARNING: This method executes actions on the host machine based on
        potentially LLM-generated input. Ensure robust validation via
        `is_action_safe` has occurred before calling this.

        Args:
            action_json: Dictionary describing the action to perform.

        Returns:
            Tuple containing:
                - bool: True if the action was successful or a no-op, False otherwise.
                - str: A message describing the result or error.
        """
        action_type = action_json.get("action")
        
        if action_type == "open_app":
            app_name = action_json.get("app_name")
            success, result_msg = self.open_app(app_name)
            
        elif action_type == "type_keystrokes":
            app_name = action_json.get("app_name")
            # Prioritize 'text' parameter, fall back to 'keys'
            keys = action_json.get("text") or action_json.get("keys")
            
            if keys is None:
                return False, "No text to type"
            
            success, result_msg = self.type_keystrokes(app_name, keys)
            
        elif action_type == "click_element":
            app_name = action_json.get("app_name")
            element_desc = action_json.get("element_desc")
            coordinates = action_json.get("coordinates")
            success, result_msg = self.click_element(app_name, element_desc, coordinates)
            
        elif action_type == "key_combo":
            app_name = action_json.get("app_name")
            combo = action_json.get("combo")
            success, result_msg = self.key_combo(app_name, combo)
            
        elif action_type == "open_url":
            browser = action_json.get("browser")
            url = action_json.get("url")
            success, result_msg = self.open_url(browser, url)

        elif action_type == "menu_click":
            app_name = action_json.get("app_name")
            menu_name = action_json.get("menu_name")
            menu_item = action_json.get("menu_item")
            success, result_msg = self.menu_click(app_name, menu_name, menu_item)

        elif action_type == "login_to_gmail":
            browser = action_json.get("browser")
            username = action_json.get("username")
            password = action_json.get("password")
            success, result_msg = self.login_to_gmail(browser, username, password)
            
        elif action_type == "none":
            # "none" action is a no-op, just return the explanation
            explanation = action_json.get("explanation", "No action needed.")
            success, result_msg = True, explanation
            
        else:
            success, result_msg = False, f"Error: Unknown action type '{action_type}'"

        # If the action was successful, add it to the action history for final summary
        if success and action_type != "none":
            history_entry = {**self.sanitize_credentials_for_display(action_json), "result": result_msg}
            self.action_history.append(history_entry)
            
        return success, result_msg

    def key_combo(self, app_name: str, combo: List[str]) -> tuple[bool, str]:
        """Execute a keyboard shortcut combo in an application."""

        # Ensure combo is a list
        if isinstance(combo, str):
            combo = [combo]

        # Convert combo to lowercase for consistent checking
        combo_lower = [c.lower() for c in combo]

        # Minimal safety check
        UNSAFE_COMBOS = {
            "command+space",    # Spotlight (system-wide)
            "command+option+esc", # Force Quit
            "command+shift+q",  # Log out
        }
        
        combo_str = "+".join(combo_lower)
        if combo_str in UNSAFE_COMBOS:
            err_msg = f"Error: Key combo '{combo_str}' is in the unsafe list"
            return False, err_msg

        if combo_str not in SAFE_KEY_COMBOS:
            err_msg = f"Error: Key combo '{combo_str}' is not in the safe list"
            return False, err_msg

        try:
            subprocess.run(['open', '-a', app_name], check=True, capture_output=True, text=True, timeout=10)
            time.sleep(0.5)  # Pause for activation

            # Parse the combo into modifiers and the key
            modifiers = [m for m in combo if m.lower() in ["command", "control", "option", "shift"]]
            keys = [k for k in combo if k.lower() not in ["command", "control", "option", "shift"]]
            
            if not keys:
                return False, "Error: No key specified in combo, only modifiers"
            
            key = keys[0]  # Get the main key
            using_clause = ""
            
            if modifiers:
                # Format modifiers for AppleScript (e.g., "using {command down, shift down}")
                modifiers_str = ", ".join([f"{m.lower()} down" for m in modifiers])
                using_clause = f"using {{{modifiers_str}}}"
            
            # Construct the key combo script
            key_combo_script = f'''
                tell application "System Events"
                    if not (UI elements enabled) then
                        return "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                    end if
                    if not (exists process "{app_name}") then
                        return "Error: Process '{app_name}' does not exist. Cannot send key combo."
                    end if

                    try
                        tell process "{app_name}"
                            set frontmost to true
                            delay 0.3
                        end tell
                        tell application "{app_name}"
                            activate
                        end tell
                        delay 0.2
                        
                        -- Execute the key combination
                        key code (ASCII character "{key}") {using_clause}
                        return "Successfully executed key combo: {combo_str}"
                    on error errMsg number errorNumber
                        return "Error during key combo: " & errMsg & " (" & errorNumber & ")"
                    end try
                end tell
            '''
            
            # For special keys like "t", "r", etc., we need to use the key name instead of ASCII; this is a common issue with AppleScript
            if len(key) == 1 and key.isalpha():
                key_combo_script = key_combo_script.replace(
                    f'key code (ASCII character "{key}") {using_clause}',
                    f'keystroke "{key}" {using_clause}'
                )
            
            # Execute the script
            result = subprocess.run(
                ['osascript', '-e', key_combo_script],
                capture_output=True,
                text=True,
                timeout=20
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # Check for errors
            if stderr:
                return False, f"Error: osascript failed for key combo. Stderr: {stderr}"

            if stdout.startswith("Error:"):
                return False, stdout
            elif "Successfully executed" in stdout:
                return True, f"Successfully executed key combo '{combo_str}' in {app_name}."
            else:
                return False, f"Unexpected result from key combo: {stdout}"

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else f"'open -a {app_name}' command failed."
            return False, f"Error activating or sending key combo to {app_name}. Details: {stderr_output}"
        except FileNotFoundError:
            return False, "Error: Required command ('osascript' or 'open') not found."
        except subprocess.TimeoutExpired:
            return False, f"Error: Timeout waiting for {app_name} to respond to key combo."
        except Exception as e:
            return False, f"Error: An unexpected error ({type(e).__name__}) occurred trying to send key combo to {app_name}."

    def open_url(self, browser: Optional[str], url: str) -> tuple[bool, str]:
        """Open a URL in the specified browser or default browser."""
        
        # Double-check URL safety
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            
            # Check protocol
            if parsed_url.scheme not in ['http', 'https']:
                return False, f"Error: URL must use http or https protocol, got: {parsed_url.scheme}"
            
            # Check domain
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
                
            if domain not in SAFE_WEBSITE_DOMAINS:
                return False, f"Error: Domain '{domain}' is not in the safe list"
                
        except Exception as e:
            return False, f"Error: Could not validate URL: {e}"

        try:
            # Determine which command to use based on whether a browser is specified
            if browser:
                # First, try to activate the specified browser
                activation_result = subprocess.run(
                    ['open', '-a', browser], 
                    check=False,  # Don't raise exception on non-zero return code
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                # Check if the browser was found
                if activation_result.returncode != 0:
                    return False, f"Error: Could not find or open browser '{browser}'"
                
                # Wait for browser to activate
                time.sleep(1)
                
                # Use AppleScript to open the URL in the specified browser
                open_url_script = f'''
                    tell application "{browser}"
                        activate
                        open location "{url}"
                        return "Successfully opened URL in {browser}."
                    end tell
                '''
            else:
                # Use the 'open' command with URL to open in default browser
                open_url_script = f'''
                    do shell script "open '{url}'"
                    return "Successfully opened URL in default browser."
                '''
                
            # Execute the script
            result = subprocess.run(
                ['osascript', '-e', open_url_script],
                capture_output=True,
                text=True,
                timeout=20
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # Check for errors
            if stderr:
                return False, f"Error: Failed to open URL. {stderr}"

            if "Successfully opened URL" in stdout:
                return True, f"Successfully opened URL '{url}' in {browser or 'default'} browser."
            else:
                return False, f"Unexpected result when opening URL: {stdout}"

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else "Command failed with no error message."
            return False, f"Error opening URL '{url}'. Details: {stderr_output}"
        except FileNotFoundError:
            return False, "Error: Required command ('osascript' or 'open') not found."
        except subprocess.TimeoutExpired:
            return False, f"Error: Timeout waiting for browser to open URL."
        except subprocess.TimeoutExpired:
            return False, f"Error: Timeout waiting for {app_name} to respond via 'open' command."
        except Exception as e:
            return False, f"Error: An unexpected error ({type(e).__name__}) occurred trying to open URL '{url}'."

    def type_keystrokes(self, app_name: str, keys: Optional[str] = None) -> tuple[bool, str]:
        """Type keystrokes into a specific application using AppleScript."""
        # Ensure keys is a string and not None
        if keys is None:
            return False, "No text to type"

        if app_name == "TextEdit":
            ensure_success, ensure_msg = self._ensure_new_textedit_document()
            if not ensure_success:
                # Failed to ensure a new document was ready
                return False, f"Failed to prepare TextEdit for typing: {ensure_msg}"

        try:
            # Ensure the target app is frontmost first using 'open -a' which also activates
            subprocess.run(['open', '-a', app_name], check=True, capture_output=True, text=True, timeout=10)
            time.sleep(0.5) # Pause for activation

            # Escape characters for AppleScript string: backslash and double quote
            escaped_keys = keys.replace('\\', '\\\\').replace('"', '\\"')

            # Construct the keystroke script using System Events
            # Added checks for UI elements enabled, may need refinement
            keystroke_script = f'''
                tell application "System Events"
                    if not (UI elements enabled) then
                        return "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                    end if
                    if not (exists process "{app_name}") then
                        return "Error: Process '{app_name}' does not exist. Cannot send keystrokes."
                    end if

                    try
                        tell process "{app_name}"
                            perform action "AXRaise" of window 1 -- Try to bring window forward reliably
                            set frontmost to true
                            delay 0.3 -- Increased delay
                        end tell
                        -- Send keystrokes to the application process itself
                        tell application "{app_name}"
                           activate -- Ensure it has focus one more time
                        end tell
                        delay 0.2
                        keystroke "{escaped_keys}"
                        return "Keystrokes sent successfully."
                    on error errMsg number errorNumber
                        return "Error during keystroke: " & errMsg & " (" & errorNumber & ")"
                    end try
                end tell
            '''
            result = subprocess.run(
                ['osascript', '-e', keystroke_script],
                capture_output=True,
                text=True,
                timeout=20 # Allow more time for typing + UI checks
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # Prioritize stderr for infrastructure errors
            if stderr:
                 if "Application can't be found" in stderr or "doesn't understand" in stderr.lower():
                      return False, f"Error: Could not find process or application '{app_name}' to send keystrokes. Is it running?"
                 if "System Events got an error: UI element scripting is not enabled" in stderr:
                      return False, "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                 return False, f"Error: osascript failed for keystrokes. Stderr: {stderr}"

            # Check stdout for AppleScript-level errors
            if stdout.startswith("Error:"):
                return False, stdout # Return the specific error from AppleScript
            elif "Keystrokes sent successfully" in stdout:
                return True, f"Successfully sent keystrokes '{keys}' to {app_name}."
            else:
                return False, f"Unexpected result sending keystrokes to {app_name}: {stdout}"

        except subprocess.CalledProcessError as e:
            # This might catch the activation failure via 'open -a'
            stderr_output = e.stderr.strip() if e.stderr else f"'open -a {app_name}' command failed."
            return False, f"Error activating or typing in {app_name}. Details: {stderr_output}"

    def extract_credentials(self, task: str, context: str) -> None:
        """
        Extract credentials from task or context using regex patterns.
        This stores credentials securely in memory for later use.
        """
        # Pattern for Gmail credentials format
        gmail_pattern = r"gmail\s+(?:credentials|login|username|account)[:=\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s+(?:password|pwd|pass)[:=\s]+([^\s,;]+)"
        
        # Combined text to search
        combined_text = f"{task}\n{context}"
        
        # Find Gmail credentials
        gmail_match = re.search(gmail_pattern, combined_text, re.IGNORECASE)
        if gmail_match:
            email = gmail_match.group(1)
            password = gmail_match.group(2)
            self.credentials_store["gmail"] = {
                "username": email,
                "password": password
            }
        
        # Add more patterns for other services as needed

    def inject_credentials(self, action: Dict[str, Any], service: str) -> Dict[str, Any]:
        """
        Inject stored credentials into action if available.
        Returns the original action if no credentials are found.
        """
        if service not in self.credentials_store:
            return action
            
        if service == "gmail" and action.get("action") == "login_to_gmail":
            # Make a copy to avoid modifying the original
            action_copy = action.copy()
            
            # Inject credentials
            creds = self.credentials_store["gmail"]
            action_copy["username"] = creds["username"]
            action_copy["password"] = creds["password"]
            
            return action_copy
            
        return action

    def sanitize_credentials_for_display(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize credentials in action objects for safe display in logs and UI.
        Returns a copy with passwords masked.
        """
        if not isinstance(action, dict):
            return action
            
        # Make a copy to avoid modifying the original
        sanitized = action.copy()
        
        # Mask password if present
        if "password" in sanitized:
            sanitized["password"] = "********"
            
        return sanitized

    def menu_click(self, app_name: str, menu_name: str, menu_item: str) -> tuple[bool, str]:
        """
        Click a menu item in an application menu bar.
        
        :param app_name: Name of the application
        :param menu_name: Name of the menu (e.g., "File", "Edit")
        :param menu_item: Name of the menu item to click (e.g., "New", "Save")
        :return: Tuple of (success, message)
        """
        
        try:
            # First, activate the application
            activation_result = subprocess.run(
                ['open', '-a', app_name],
                check=False,  # Don't raise exception on non-zero return code
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Check if the app was found
            if activation_result.returncode != 0:
                return False, f"Error: Could not find or open application '{app_name}'"
            
            # Wait for app to activate
            time.sleep(1)
            
            # Construct AppleScript to click the menu item
            menu_click_script = f'''
                tell application "System Events"
                    if not (UI elements enabled) then
                        return "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                    end if
                    
                    if not (exists process "{app_name}") then
                        return "Error: Process '{app_name}' does not exist. Cannot click menu item."
                    end if
                    
                    try
                        tell process "{app_name}"
                            set frontmost to true
                            delay 0.5
                            
                            -- Try to click the menu item
                            tell menu bar 1
                                tell menu bar item "{menu_name}"
                                    tell menu "{menu_name}"
                                        click menu item "{menu_item}"
                                        return "Successfully clicked menu item '{menu_item}' in menu '{menu_name}'."
                                    end tell
                                end tell
                            end tell
                        end tell
                    on error errMsg number errorNumber
                        return "Error clicking menu item: " & errMsg & " (" & errorNumber & ")"
                    end try
                end tell
            '''
            
            # Execute the script
            result = subprocess.run(
                ['osascript', '-e', menu_click_script],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            # Check for errors
            if stderr:
                if "System Events got an error: UI element scripting is not enabled" in stderr:
                    return False, "Error: UI element scripting is not enabled. Please enable it in System Settings > Privacy & Security > Accessibility."
                return False, f"Error: osascript failed for menu click. Stderr: {stderr}"
            
            # Check stdout for AppleScript-level errors
            if stdout.startswith("Error:"):
                return False, stdout
            elif "Successfully clicked menu item" in stdout:
                return True, f"Successfully clicked menu '{menu_name}' > '{menu_item}' in '{app_name}'"
            else:
                return False, f"Unexpected result from menu click: {stdout}"
                
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else "Command failed with no error message."
            return False, f"Error clicking menu '{menu_name}' > '{menu_item}' in {app_name}. Details: {stderr_output}"
        except FileNotFoundError:
            return False, "Error: Required command ('osascript' or 'open') not found."
        except subprocess.TimeoutExpired:
            return False, f"Error: Timeout waiting for menu click operation in {app_name}."
        except Exception as e:
            return False, f"Error: An unexpected error ({type(e).__name__}) occurred trying to click menu in {app_name}."

    def login_to_gmail(self, browser: str, username: str, password: str) -> tuple[bool, str]:
        """
        Log in to Gmail using the specified browser and credentials.
        
        :param browser: Browser to use (e.g., "Safari", "Chrome", "Firefox", "Google Chrome")
        :param username: Gmail username/email
        :param password: Gmail password
        :return: Tuple of (success, message)
        """
        # Validate input parameters
        if not username or not isinstance(username, str):
            return False, "Error: Invalid username provided for Gmail login"
        
        if not password or not isinstance(password, str):
            return False, "Error: Invalid password provided for Gmail login"
        
        # Normalize browser name
        if browser.lower() == "google chrome":
            browser = "Google Chrome"
        
        try:
            # 1. Open Gmail in the specified browser
            gmail_url = "https://mail.google.com"
            open_success, open_msg = self.open_url(browser, gmail_url)
            
            if not open_success:
                return False, f"Failed to open Gmail: {open_msg}"
            
            # Allow time for page to load
            time.sleep(3)
            
            # 2. Check if we need to sign in or if already signed in
            check_login_script = f'''
                tell application "System Events"
                    if not (UI elements enabled) then
                        return "Error: UI element scripting is not enabled."
                    end if
                    
                    if not (exists process "{browser}") then
                        return "Error: Browser not running."
                    end if
                    
                    tell process "{browser}"
                        set frontmost to true
                        delay 1
                        
                        -- Check for either login form or already logged in state
                        set loginPageElements to 0
                        set gmailElements to 0
                        
                        -- Check for email input field (sign-in page indicator)
                        try
                            set emailFields to (text fields whose name contains "Email" or name contains "identifier" or name contains "username")
                            set loginPageElements to count of emailFields
                        end try
                        
                        -- Check for Gmail interface elements (already logged in indicator)
                        try
                            set gmailElements to count of (UI elements whose role is "button" and name contains "Compose")
                        end try
                        
                        if gmailElements > 0 then
                            return "already_logged_in"
                        else if loginPageElements > 0 then
                            return "need_login"
                        else
                            return "loading"
                        end if
                    end tell
                end tell
            '''
            
            # Execute login check script
            check_result = subprocess.run(
                ['osascript', '-e', check_login_script],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            check_output = check_result.stdout.strip()
            
            if check_output == "already_logged_in":
                return True, "Already logged in to Gmail"
                
            elif check_output == "loading":
                time.sleep(5)  # Wait longer for page to load
                
            # 3. Perform the login sequence
            login_script = f'''
                tell application "System Events"
                    tell process "{browser}"
                        set frontmost to true
                        delay 1
                        
                        -- Look for email input field
                        set emailField to text field 1
                        set focused of emailField to true
                        delay 0.5
                        keystroke "{username}"
                        delay 1
                        
                        -- Look for "Next" button and click it
                        set nextButton to (buttons whose name contains "Next" or name contains "Continue")
                        if (count of nextButton) > 0 then
                            click item 1 of nextButton
                        else
                            -- Try pressing Enter instead
                            keystroke return
                        end if
                        
                        -- Wait for password field to appear
                        delay 2
                        
                        -- Enter password
                        set passwordField to text field 1
                        set focused of passwordField to true
                        delay 0.5
                        keystroke "{password}"
                        delay 1
                        
                        -- Click sign in button
                        set signInButton to (buttons whose name contains "Sign in" or name contains "Next" or name contains "Log in")
                        if (count of signInButton) > 0 then
                            click item 1 of signInButton
                        else
                            -- Try pressing Enter instead
                            keystroke return
                        end if
                        
                        return "Login sequence completed"
                    end tell
                end tell
            '''
            
            # Execute login script
            login_result = subprocess.run(
                ['osascript', '-e', login_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            login_output = login_result.stdout.strip()
            login_error = login_result.stderr.strip()
            
            if login_error:
                return False, f"Error during Gmail login: {login_error}"
            
            # 4. Verify successful login by checking for Gmail interface elements
            time.sleep(5)  # Wait for redirect and page load after login
            
            verify_script = f'''
                tell application "System Events"
                    tell process "{browser}"
                        set frontmost to true
                        delay 1
                        
                        -- Check for Gmail interface elements (compose button, inbox, etc.)
                        try
                            set composeButton to (buttons whose name contains "Compose")
                            if (count of composeButton) > 0 then
                                return "login_successful"
                            end if
                        end try
                        
                        -- Check for other Gmail elements
                        try
                            set inboxElement to (UI elements whose name contains "Inbox")
                            if (count of inboxElement) > 0 then
                                return "login_successful"
                            end if
                        end try
                        
                        return "verification_failed"
                    end tell
                end tell
            '''
            
            # Execute verification script
            verify_result = subprocess.run(
                ['osascript', '-e', verify_script],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            verify_output = verify_result.stdout.strip()
            
            if verify_output == "login_successful":
                return True, "Successfully logged in to Gmail"
            else:
                return False, "Could not verify successful login to Gmail"
            
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else "Command failed with no error message"
            return False, f"Error during Gmail login process: {stderr_output}"
        except Exception as e:
            return False, f"Error: An unexpected error ({type(e).__name__}) occurred during Gmail login"