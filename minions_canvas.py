import streamlit as st
import time
from typing import List, Dict, Any, Optional
import re

from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.perplexity import PerplexityAIClient
from minions.minion import Minion


class StreamlitMinionsCanvas:
    def __init__(self):
        st.set_page_config(
            page_title="Minions Canvas",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize state if not already done
        if "initialized" not in st.session_state:
            self.initialize_state()

        # Setup UI
        self.create_ui()

    def initialize_state(self):
        """Initialize the session state variables"""
        st.session_state.initialized = True
        st.session_state.current_round = 0
        st.session_state.max_rounds = 2
        st.session_state.messages_history = []
        st.session_state.waiting_for_edit = False
        st.session_state.current_role = None
        st.session_state.process_running = False
        st.session_state.process_complete = False
        st.session_state.edited_message = None
        st.session_state.final_output = None
        st.session_state.error = None

        # Initialize clients
        self.setup_minion()

    def setup_minion(self):
        """Initialize the minion clients"""
        # Store model selections in session state
        if "worker_model" not in st.session_state:
            st.session_state.worker_model = "llama3.2"
        if "supervisor_model" not in st.session_state:
            st.session_state.supervisor_model = "gpt-4o"
        if "supervisor_provider" not in st.session_state:
            st.session_state.supervisor_provider = "openai"

    def create_ui(self):
        """Create the main UI components"""
        # Sidebar for configuration, input and message history
        st.title("Minions Canvas")
        st.markdown(
            "An interactive canvas for editing and visualizing minion communications"
        )
        with st.sidebar:

            st.header("Configuration")

            # Model selection
            st.session_state.worker_model = st.selectbox(
                "Worker Model", ["llama3.2", "llama3", "mistral", "mixtral"], index=0
            )

            # Add provider selection for supervisor
            st.session_state.supervisor_provider = st.selectbox(
                "Supervisor Provider", ["openai", "perplexity"], index=0
            )

            # Model options based on provider
            if st.session_state.supervisor_provider == "openai":
                model_options = [
                    "gpt-4o",
                    "gpt-3.5-turbo",
                    "claude-3-opus",
                    "claude-3-sonnet",
                ]
                default_index = 0
            else:  # perplexity
                model_options = [
                    "sonar-reasoning",
                    "sonar-reasoning-pro",
                ]
                default_index = 1  # Default to sonar-large-online

            st.session_state.supervisor_model = st.selectbox(
                "Supervisor Model", model_options, index=default_index
            )

            st.session_state.max_rounds = st.slider(
                "Maximum Rounds", min_value=1, max_value=5, value=2
            )

            # Reset button
            if st.button("Reset"):
                self.reset_state()

            # Input section moved to sidebar
            st.header("Input")

            # Task and context inputs
            task = st.text_area("Task", height=100, key="task_input")
            context = st.text_area("Context", height=150, key="context_input")

            # Start button
            start_button = st.button(
                "Start Process",
                disabled=st.session_state.process_running,
                use_container_width=True,
            )

            if start_button:
                self.start_minion(task, context)

            # Message History moved to sidebar
            st.header("Message History")
            for i, msg in enumerate(st.session_state.messages_history):
                with st.expander(f"{msg['role']} (Round {msg['round']})"):
                    # Render message content as markdown
                    st.markdown(msg["content"])

        # Display current state with enhanced styling
        if st.session_state.process_running and st.session_state.waiting_for_edit:
            st.markdown(
                f"<div class='canvas-container'><h3>Editing {st.session_state.current_role}</h3></div>",
                unsafe_allow_html=True,
            )

            # Add formatting toolbar for the edit area
            format_col1, format_col2, format_col3, format_col4 = st.columns(
                [1, 1, 1, 1]
            )

            with format_col1:
                wrap = st.checkbox("Word Wrap", value=True)

            with format_col2:
                line_numbers = st.checkbox("Line Numbers", value=False)

            with format_col3:
                syntax_highlight = st.checkbox("Syntax Highlighting", value=True)

            with format_col4:
                render_markdown = st.checkbox("Preview Markdown", value=False)

            with format_col4:
                if st.button("Copy to Clipboard", use_container_width=True):
                    st.write("Content copied to clipboard!")

            # Editable text area with enhanced styling
            edited_message = st.text_area(
                "",  # No label, we use the header above
                value=st.session_state.current_message,
                height=600,
                key="edit_area",
                help="Edit the content here. Use the formatting options above to customize the view.",
            )

            # Add line numbers if enabled
            if line_numbers:
                lines = edited_message.split("\n")
                numbered_text = "\n".join(
                    [f"{i+1}: {line}" for i, line in enumerate(lines)]
                )
                st.code(numbered_text, language="python" if syntax_highlight else None)

            # Preview markdown rendering if enabled
            if render_markdown:
                st.markdown("### Markdown Preview")
                st.markdown(edited_message)

            # Action buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Proceed", use_container_width=True):
                    self.proceed_with_edited_message(edited_message)
                    st.rerun()

            with col2:
                if st.button("Reset Content", use_container_width=True):
                    st.session_state.current_message = (
                        st.session_state.messages_history[-1]["content"]
                    )
                    st.rerun()

        elif st.session_state.process_complete:
            st.markdown("<div class='canvas-container'>", unsafe_allow_html=True)
            st.success("Process completed!")
            st.markdown("### Final Output:")

            # Add export options for final output
            export_col1, export_col2, export_col3 = st.columns([1, 1, 1])

            with export_col1:
                if st.button("Export as Markdown"):
                    self.export_content(format="md")

            with export_col2:
                if st.button("Export as Text"):
                    self.export_content(format="txt")

            with export_col3:
                if st.button("Copy Output"):
                    st.write("Output copied to clipboard!")

            st.markdown(st.session_state.final_output)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.error:
            st.error(f"Error: {st.session_state.error}")

        elif st.session_state.process_running:
            st.info("Processing...")
            st.spinner()

        else:
            st.info("Enter a task and context, then click 'Start Process'")

    def start_minion(self, task, context):
        """Start the minion process"""
        if not task:
            st.error("Task cannot be empty")
            return

        # Reset state
        st.session_state.current_round = 0
        st.session_state.messages_history = []
        st.session_state.process_running = True
        st.session_state.process_complete = False
        st.session_state.final_output = None
        st.session_state.error = None

        # Initialize clients
        try:
            worker_client = OllamaClient(model_name=st.session_state.worker_model)

            # Create supervisor client based on provider selection
            if st.session_state.supervisor_provider == "openai":
                supervisor_client = OpenAIClient(
                    model_name=st.session_state.supervisor_model
                )
            else:  # perplexity
                supervisor_client = PerplexityAIClient(
                    model_name=st.session_state.supervisor_model
                )

            # Create custom minion
            custom_minion = StreamlitCustomMinion(worker_client, supervisor_client)

            # Start the process in a separate thread
            st.session_state.task = task
            st.session_state.context = context
            st.session_state.custom_minion = custom_minion

            # Start the first step
            self.run_next_step()

        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.process_running = False

    def run_next_step(self):
        """Run the next step in the minion process"""
        custom_minion = st.session_state.custom_minion

        try:
            # Determine which step to run next
            if not hasattr(st.session_state, "current_step"):
                st.session_state.current_step = "worker_prompt"
                st.session_state.worker_response = None
                st.session_state.messages = []

            if st.session_state.current_step == "worker_prompt":
                # Create worker prompt
                worker_messages = custom_minion.create_worker_prompt(
                    st.session_state.task,
                    st.session_state.context,
                    st.session_state.worker_response,
                )

                # Store the messages for later use
                st.session_state.messages = worker_messages
                # Display for editing
                st.session_state.waiting_for_edit = True
                st.session_state.current_role = "worker prompt"
                st.session_state.current_message = worker_messages[-1]["content"]

                # Add to history without waiting for edit
                st.session_state.messages_history.append(
                    {
                        "role": "Worker Prompt",
                        "content": worker_messages[-1]["content"],
                        "round": st.session_state.current_round,
                    }
                )

                st.session_state.next_step = "worker_response"

            elif st.session_state.current_step == "worker_response":
                # Get worker response
                worker_response, _ = custom_minion.process_worker_response(
                    st.session_state.messages
                )

                # Display for editing
                st.session_state.waiting_for_edit = True
                st.session_state.current_role = "worker response"
                st.session_state.current_message = worker_response

                # Add to history
                st.session_state.messages_history.append(
                    {
                        "role": "Worker Response",
                        "content": worker_response,
                        "round": st.session_state.current_round,
                    }
                )

                # Store for later use
                st.session_state.worker_response = worker_response

                # Next step
                st.session_state.next_step = "supervisor_prompt"

            elif st.session_state.current_step == "supervisor_prompt":
                # Create supervisor prompt using the worker response
                supervisor_messages = custom_minion.create_supervisor_prompt(
                    st.session_state.task,
                    st.session_state.context,
                    st.session_state.worker_response,  # Use the worker response directly
                )

                st.session_state.waiting_for_edit = True
                st.session_state.current_role = "supervisor prompt"
                st.session_state.current_message = supervisor_messages[-1]["content"]

                # Store the messages for later use
                st.session_state.messages = supervisor_messages

                # Add to history without waiting for edit
                st.session_state.messages_history.append(
                    {
                        "role": "Supervisor Prompt",
                        "content": supervisor_messages[-1]["content"],
                        "round": st.session_state.current_round,
                    }
                )

                st.session_state.next_step = "supervisor_response"

            elif st.session_state.current_step == "supervisor_response":
                # Get supervisor response
                supervisor_response, _ = custom_minion.process_supervisor_response(
                    st.session_state.messages
                )

                # Display for editing
                st.session_state.waiting_for_edit = True
                st.session_state.current_role = "supervisor response"
                st.session_state.current_message = supervisor_response

                # Add to history
                st.session_state.messages_history.append(
                    {
                        "role": "Supervisor Response",
                        "content": supervisor_response,
                        "round": st.session_state.current_round,
                    }
                )

                # Next step
                st.session_state.next_step = "check_continue"

            elif st.session_state.current_step == "check_continue":
                # Use the edited supervisor response
                supervisor_response = st.session_state.edited_message

                # Check if we should continue
                if (
                    custom_minion.should_continue(supervisor_response)
                    and st.session_state.current_round < st.session_state.max_rounds - 1
                ):
                    # Increment round and continue
                    st.session_state.current_round += 1
                    st.session_state.next_step = "worker_prompt"
                else:
                    # Complete the process
                    st.session_state.process_complete = True
                    st.session_state.process_running = False
                    st.session_state.final_output = st.session_state.current_message
                    st.session_state.waiting_for_edit = False
                    return

            # Update current step for next iteration
            st.session_state.current_step = st.session_state.next_step

        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.process_running = False
            st.session_state.waiting_for_edit = False

    def proceed_with_edited_message(self, edited_message):
        """Process the edited message and continue"""
        if not st.session_state.waiting_for_edit:
            return

        # Save the edited message
        st.session_state.edited_message = edited_message

        # Update the message in history
        if st.session_state.messages_history:
            st.session_state.messages_history[-1]["content"] = edited_message

        # Update the actual messages that will be sent to the models
        if st.session_state.current_role == "worker prompt":
            # Update the last message in the worker prompt messages
            st.session_state.messages[-1]["content"] = edited_message
        elif st.session_state.current_role == "supervisor prompt":
            # Update the last message in the supervisor prompt messages
            st.session_state.messages[-1]["content"] = edited_message
        elif st.session_state.current_role == "worker response":
            # Store the edited worker response
            st.session_state.worker_response = edited_message
        elif st.session_state.current_role == "supervisor response":
            # For supervisor response, we just need to store it for the next check_continue step
            pass

        # Reset waiting state
        st.session_state.waiting_for_edit = False

        # Run the next step
        self.run_next_step()

    def reset_state(self):
        """Reset the application state"""
        for key in list(st.session_state.keys()):
            if key not in ["worker_model", "supervisor_model", "supervisor_provider"]:
                del st.session_state[key]

        self.initialize_state()

    def export_content(self, format="md"):
        """Export the content to a file"""
        if st.session_state.process_complete and st.session_state.final_output:
            content = st.session_state.final_output
            filename = f"minions_output.{format}"

            # Create a download link
            st.download_button(
                label=f"Download as {format.upper()}",
                data=content,
                file_name=filename,
                mime=f"text/{format}",
            )
        else:
            st.warning("No content to export yet. Complete the process first.")


class StreamlitCustomMinion(Minion):
    """Custom Minion class for Streamlit integration that allows editing messages at each step"""

    def __init__(self, worker_client, supervisor_client):
        # Store the clients directly as attributes
        self.worker_client = worker_client
        self.supervisor_client = supervisor_client
        # Call the parent constructor
        super().__init__(worker_client, supervisor_client)

    def __call__(self, task, context=None, max_rounds=1):
        """Override the call method to intercept messages at each step for editing"""
        # This method will be called by the StreamlitMinionsCanvas class
        # We'll use the parent class's implementation but intercept at key points

        # Store these for use in the step-by-step process
        self.task = task
        self.context = context or []
        self.max_rounds = max_rounds

        # The actual execution will be handled by the StreamlitMinionsCanvas class
        # which will call the appropriate methods at each step
        return None

    def create_worker_prompt(self, task, context, previous_response=None):
        """Create a prompt for the worker model (public wrapper for _create_worker_prompt)"""
        # Join context sections if it's a list
        if isinstance(context, list):
            context_text = "\n\n".join(context)
        else:
            context_text = context

        # Create system message with context and task
        system_message = {
            "role": "system",
            "content": f"Context:\n{context_text}\n\nTask: {task}",
        }

        # Create user message based on whether this is a follow-up
        if previous_response is not None:
            user_message = {
                "role": "user",
                "content": f"Please revise your previous response based on this feedback: {previous_response}",
            }
        else:
            user_message = {
                "role": "user",
                "content": f"Be creative and attempt to complete the task!",
            }

        return [system_message, user_message]

    def create_supervisor_prompt(self, task, context, worker_response):
        """Create a prompt for the supervisor model (public wrapper for internal methods)"""
        # For the supervisor, we'll use the SUPERVISOR_CONVERSATION_PROMPT from the parent class
        from minions.prompts.minion import SUPERVISOR_CONVERSATION_PROMPT

        prompt = SUPERVISOR_CONVERSATION_PROMPT.format(response=worker_response)

        return [{"role": "user", "content": prompt}]

    def process_worker_response(self, messages):
        """Process the worker's response"""
        response, usage, _ = self.worker_client.chat(messages=messages)
        # Remove any <think>...</think> blocks from the response
        cleaned_response = self.remove_think_blocks(response[0])
        return cleaned_response, usage

    def remove_think_blocks(self, text):
        """Remove any content between <think> and </think> tags"""
        # Use regex to remove all <think>...</think> blocks
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_text

    def process_supervisor_response(self, messages):
        """Process the supervisor's response"""
        response, usage = self.supervisor_client.chat(messages=messages)
        cleaned_response = self.remove_think_blocks(response[0])
        return cleaned_response, usage

    def should_continue(self, supervisor_response):
        """Determine if we should continue the conversation based on supervisor response"""
        # Check if the supervisor wants to continue
        return "CONTINUE" in supervisor_response


if __name__ == "__main__":
    app = StreamlitMinionsCanvas()
