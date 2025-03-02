import streamlit as st
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.together import TogetherClient

# Prompts moved from gateway.py
WORKER_SYSTEM_PROMPT = """\
You are a prompt engineer tasked with refining and reformatting user prompts for optimal use by a powerful reasoning model such as OpenAI's o1, o3, or DeepSeek's r1.
"""

WORKER_USER_PROMPT = """\
You will receive a draft prompt from the user. Your goal is to refine it using best practices for reasoning model prompts. Specifically:

1. Structure the prompt clearly using labeled sections:
   - Task: Clearly state the problem to be solved.
   - Context (if needed): Include only essential background information.
   - Output Format: Define how the response should be structured.

2. Enhance clarity and conciseness by removing unnecessary details while preserving all relevant constraints and requirements.

3. Use a structured and minimalistic approach to let the model leverage its internal reasoning ability without excessive guidance.

4. Ensure correctness and completeness by verifying that all necessary details (such as input constraints and expected outputs) are retained.

5. Avoid over-specification of reasoning steps unless explicitly required.

Example Input:
"I want a list of the best medium-length hikes within two hours of San Francisco. Each hike should provide a cool and unique adventure, and be lesser known. For each hike, return the name as listed on AllTrails, the starting and ending addresses, the distance, drive time, hike duration, and a brief explanation of what makes it unique. Return the top three. Ensure the trail names and details are accurate."

Example Output:
Task: Identify three of the best medium-length hikes within two hours of San Francisco that provide a unique adventure and are relatively lesser known.

Context: The hikes should not be commonly recommended within the city (e.g., Golden Gate Park or Presidio). The user prefers trails with scenic ocean views and an interesting endpoint (such as a restaurant or historical site).

Output Format:
- Provide a ranked list of the top three hikes.
- For each hike, include:
  - Official trail name (as listed on AllTrails)
  - Starting and ending addresses
  - Distance, estimated drive time, and hike duration
  - A brief explanation of what makes the hike unique

Ensure that all provided details are accurate and that each hike is correctly named and verifiable.

User Prompt: {task}

Please refine the prompt for a powerful reasoning model. Ensure that the prompt is clear, specific, and actionable.

Below your prompt, we will concatenate any additional raw context provided by the user.
"""

SUPERVISOR_PROMPT = """\
{refined_input}

Additional Context (if provided):
{context}
"""

import os
import time
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import io
from streamlit_theme import st_theme

# Set custom sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 350px;
            max-width: 750px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# OpenAI model pricing per 1M tokens
OPENAI_PRICES = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "o1": {"input": 5.00, "cached_input": 2.50, "output": 15.00},
}

PROVIDER_TO_ENV_VAR_KEY = {
    "OpenAI": "OPENAI_API_KEY",
    "Together": "TOGETHER_API_KEY",
}

placeholder_messages = {}

# Initialize session state variables
if "refined_prompt" not in st.session_state:
    st.session_state["refined_prompt"] = ""
if "show_refinement_results" not in st.session_state:
    st.session_state["show_refinement_results"] = False
if "worker_usage" not in st.session_state:
    st.session_state["worker_usage"] = None


def is_dark_mode():
    theme = st_theme()
    if theme and "base" in theme:
        if theme["base"] == "dark":
            return True
    return False


def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None


def extract_text_from_image(image_bytes):
    """Extract text from an image file using pytesseract OCR."""
    try:
        import pytesseract

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def message_callback(role, message, is_final=True):
    """Show messages for the Gateway protocol,
    labeling the local vs remote model clearly."""
    # Map supervisor -> Remote, worker -> Local
    if role == "supervisor":
        chat_role = "Remote"
        path = "assets/gru.jpg"
    else:
        chat_role = "Local"
        path = "assets/minion.png"

    # If we are not final, render a placeholder.
    if not is_final:
        # Create a placeholder container and store it for later update.
        placeholder = st.empty()
        with placeholder.chat_message(chat_role, avatar=path):
            if role == "supervisor":
                st.markdown("**Remote model working...**")
                # Create a minimalistic progress indicator
                progress_placeholder = st.empty()
                progress_placeholder.progress(50, text="Processing your request...")
            else:
                st.markdown("**Local model refining prompt...**")
                # Create a minimalistic progress indicator
                progress_placeholder = st.empty()
                progress_placeholder.progress(50, text="Optimizing your prompt...")
        placeholder_messages[role] = placeholder
    else:
        if role in placeholder_messages:
            placeholder_messages[role].empty()
            del placeholder_messages[role]
        with st.chat_message(chat_role, avatar=path):
            if isinstance(message, dict):
                if "content" in message:
                    message_content = message["content"].replace("$", "\\$")
                    st.markdown(message_content)
                else:
                    st.write(message)
            else:
                message = message.replace("$", "\\$")
                st.markdown(message)


def initialize_clients(
    local_model_name,
    remote_model_name,
    provider,
    local_temperature,
    local_max_tokens,
    remote_temperature,
    remote_max_tokens,
    api_key,
    num_ctx=4096,
):
    """Initialize the local and remote clients for the Gateway protocol."""
    # Store model parameters in session state for potential reinitialization
    st.session_state.local_model_name = local_model_name
    st.session_state.remote_model_name = remote_model_name
    st.session_state.local_temperature = local_temperature
    st.session_state.local_max_tokens = local_max_tokens
    st.session_state.remote_temperature = remote_temperature
    st.session_state.remote_max_tokens = remote_max_tokens
    st.session_state.provider = provider
    st.session_state.api_key = api_key
    st.session_state.callback = message_callback

    # Initialize local client (Ollama)
    st.session_state.local_client = OllamaClient(
        model_name=local_model_name,
        temperature=local_temperature,
        max_tokens=int(local_max_tokens),
        num_ctx=num_ctx,
        structured_output_schema=None,
        use_async=False,
    )

    # Initialize remote client based on provider
    if provider == "OpenAI":
        st.session_state.remote_client = OpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Together":
        st.session_state.remote_client = TogetherClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    else:  # Default to OpenAI
        st.session_state.remote_client = OpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )

    # Initialize Gateway
    st.session_state.method = Gateway(
        st.session_state.local_client,
        st.session_state.remote_client,
        callback=message_callback,
    )

    return (
        st.session_state.local_client,
        st.session_state.remote_client,
        st.session_state.method,
    )


def run_gateway(task, context, doc_metadata, status, use_refined_prompt):
    """Run the Gateway protocol with pre-initialized clients."""
    setup_start_time = time.time()

    with status.container():
        messages_container = st.container()
        st.markdown(f"**Query:** {task}")

        # Adjust context window if needed
        if "local_client" in st.session_state and hasattr(
            st.session_state.local_client, "num_ctx"
        ):
            padding = 8000
            estimated_tokens = int(len(context) / 4 + padding) if context else 4096
            num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
            closest_value = min(
                [x for x in num_ctx_values if x >= estimated_tokens], default=131072
            )

            # Only reinitialize if num_ctx needs to change
            if closest_value != st.session_state.local_client.num_ctx:
                st.write(f"Adjusting context window to {closest_value} tokens...")

                # Reinitialize the local client with the new num_ctx
                if (
                    "local_model_name" in st.session_state
                    and "local_temperature" in st.session_state
                    and "local_max_tokens" in st.session_state
                    and "api_key" in st.session_state
                ):
                    # Reinitialize the local client with the new num_ctx
                    st.session_state.local_client = OllamaClient(
                        model_name=st.session_state.local_model_name,
                        temperature=st.session_state.local_temperature,
                        max_tokens=int(st.session_state.local_max_tokens),
                        num_ctx=closest_value,
                        structured_output_schema=None,
                        use_async=False,
                    )

                    # Reinitialize the method with the new local client
                    st.session_state.method = Gateway(
                        st.session_state.local_client,
                        st.session_state.remote_client,
                        callback=message_callback,
                    )

        setup_time = time.time() - setup_start_time
        st.write("Processing query...")
        execution_start_time = time.time()

        # If use_refined_prompt is False, bypass the worker and directly call the supervisor
        if not use_refined_prompt:
            # Create a direct supervisor message
            supervisor_messages = [
                {
                    "role": "user",
                    "content": f"{task}\n\nAdditional Context (if provided):\n{context}",
                }
            ]

            # Call the supervisor directly
            if message_callback:
                message_callback("supervisor", None, is_final=False)

            supervisor_response, supervisor_usage = st.session_state.remote_client.chat(
                messages=supervisor_messages
            )

            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            if message_callback:
                message_callback("supervisor", supervisor_messages[-1])

            # Import Usage class
            from minions.usage import Usage

            # Create a similar output structure to the Gateway protocol
            output = {
                "final_answer": supervisor_response[0],
                "supervisor_messages": supervisor_messages,
                "worker_messages": [],
                "remote_usage": supervisor_usage,
                "local_usage": Usage(
                    completion_tokens=0, prompt_tokens=0
                ),  # Zero tokens instead of None
            }
        else:
            # Use the normal Gateway protocol
            output = st.session_state.method(
                task=task,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=1,
            )

        execution_time = time.time() - execution_start_time

    return output, setup_time, execution_time


def validate_openai_key(api_key):
    try:
        client = OpenAIClient(
            model_name="gpt-4o-mini", api_key=api_key, temperature=0.0, max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_together_key(api_key):
    try:
        client = TogetherClient(
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


# Check theme setting
dark_mode = is_dark_mode()

# Choose image based on theme
if dark_mode:
    image_path = "assets/minions_logo_no_background.png"
else:
    image_path = "assets/minions_logo_light.png"

# Display Minions logo at the top
st.image(image_path, use_container_width=True)

# add a horizontal line that is width of image
st.markdown("<hr style='width: 100%;'>", unsafe_allow_html=True)

# ---------------------------
#  Sidebar for LLM settings
# ---------------------------
with st.sidebar:
    st.subheader("LLM Provider Settings")

    provider_col, key_col = st.columns([1, 2])
    with provider_col:
        # Only offer OpenAI and Together as providers
        providers = ["OpenAI", "Together"]
        selected_provider = st.selectbox(
            "Select LLM provider",
            options=providers,
            index=0,
        )

    env_var_name = f"{selected_provider.upper()}_API_KEY"
    env_key = os.getenv(env_var_name)
    with key_col:
        user_key = st.text_input(
            f"{selected_provider} API Key (optional if set in environment)",
            type="password",
            value="",
            key=f"{selected_provider}_key",
        )
    api_key = user_key if user_key else env_key

    if api_key:
        if selected_provider == "OpenAI":
            is_valid, msg = validate_openai_key(api_key)
        elif selected_provider == "Together":
            is_valid, msg = validate_together_key(api_key)
        else:
            raise ValueError(f"Invalid provider: {selected_provider}")

        if is_valid:
            st.success("**✓ Valid API key.** You're good to go!")
            provider_key = api_key
        else:
            st.error(f"**✗ Invalid API key.** {msg}")
            provider_key = None
    else:
        st.error(
            f"**✗ Missing API key.** Input your key above or set the environment variable with `export {PROVIDER_TO_ENV_VAR_KEY[selected_provider]}=<your-api-key>`"
        )
        provider_key = None

    # Model Settings
    st.subheader("Model Settings")

    # Create two columns for local and remote model settings
    local_col, remote_col = st.columns(2)

    # Local model settings
    with local_col:
        st.markdown("### Local Model")
        st.image("assets/minion_resized.jpg", use_container_width=True)
        local_model_options = {
            "llama3.2 (Recommended)": "llama3.2",
            "llama3.1:8b": "llama3.1:8b",
            "qwen2.5:3b": "qwen2.5:3b",
            "qwen2.5:7b": "qwen2.5:7b",
            "deepseek-r1:1.5b": "deepseek-r1:1.5b",
            "deepseek-r1:7b": "deepseek-r1:7b",
        }
        local_model_display = st.selectbox(
            "Model", options=list(local_model_options.keys()), index=0
        )
        local_model_name = local_model_options[local_model_display]

        show_local_params = st.toggle(
            "Change defaults", value=False, key="local_defaults_toggle"
        )
        if show_local_params:
            local_temperature = st.slider(
                "Temperature", 0.0, 2.0, 0.0, 0.05, key="local_temp"
            )
            local_max_tokens_str = st.text_input(
                "Max tokens per turn", "4096", key="local_tokens"
            )
            try:
                local_max_tokens = int(local_max_tokens_str)
            except ValueError:
                st.error("Local Max Tokens must be an integer.")
                st.stop()
        else:
            local_temperature = 0.0
            local_max_tokens = 4096

    # Remote model settings
    with remote_col:
        st.markdown("### Remote Model")
        st.image("assets/gru_resized.jpg", use_container_width=True)
        if selected_provider == "OpenAI":
            model_mapping = {
                "o1": "o1",
                "o3-mini (Recommended)": "o3-mini",
                "gpt-4o": "gpt-4o",
                "gpt-4o-mini": "gpt-4o-mini",
            }
            default_model_index = 1
        elif selected_provider == "Together":
            model_mapping = {
                "DeepSeek-R1 (Recommended)": "deepseek-ai/DeepSeek-R1",
                "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
                "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            }
            default_model_index = 0
        else:
            model_mapping = {}
            default_model_index = 0

        remote_model_display = st.selectbox(
            "Model",
            options=list(model_mapping.keys()),
            index=default_model_index,
            key="remote_model",
        )
        remote_model_name = model_mapping[remote_model_display]

        show_remote_params = st.toggle(
            "Change defaults", value=False, key="remote_defaults_toggle"
        )
        if show_remote_params:
            remote_temperature = st.slider(
                "Temperature", 0.0, 2.0, 0.0, 0.05, key="remote_temp"
            )
            remote_max_tokens_str = st.text_input(
                "Max Tokens", "4096", key="remote_tokens"
            )
            try:
                remote_max_tokens = int(remote_max_tokens_str)
            except ValueError:
                st.error("Remote Max Tokens must be an integer.")
                st.stop()
        else:
            remote_temperature = 0.0
            remote_max_tokens = 4096

# -------------------------
#   Main app layout
# -------------------------
st.subheader("Gateway to Reasoning Models")

# Single input for query
if "refined_prompt" in st.session_state and st.session_state["refined_prompt"]:
    user_query = st.text_area(
        "Enter your query or request here",
        value=st.session_state["refined_prompt"],
        height=400,
        key="user_query",
    )
else:
    user_query = st.text_area(
        "Enter your query or request here", value="", height=150, key="user_query"
    )

# File upload for context
uploaded_files = st.file_uploader(
    "Optionally upload PDF / TXT files for additional context",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

file_content = ""
if uploaded_files:
    all_file_contents = []
    total_size = 0
    file_names = []
    for uploaded_file in uploaded_files:
        try:
            file_type = uploaded_file.name.lower().split(".")[-1]
            current_content = ""
            file_names.append(uploaded_file.name)

            if file_type == "pdf":
                current_content = extract_text_from_pdf(uploaded_file.read()) or ""
            else:
                current_content = uploaded_file.getvalue().decode()

            if current_content:
                all_file_contents.append("\n--------------------")
                all_file_contents.append(
                    f"### Content from {uploaded_file.name}:\n{current_content}"
                )
                total_size += uploaded_file.size
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

    if all_file_contents:
        file_content = "\n".join(all_file_contents)
        # Create doc_metadata string
        doc_metadata = f"Input: {len(file_names)} documents ({', '.join(file_names)}). Total extracted text length: {len(file_content)} characters."
    else:
        doc_metadata = ""
else:
    doc_metadata = ""

# Combine query and file content as context
context = file_content

# Estimate token count for context window sizing
padding = 8000
estimated_tokens = int(len(context) / 4 + padding) if context else 4096
num_ctx_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
closest_value = min(
    [x for x in num_ctx_values if x >= estimated_tokens], default=131072
)
num_ctx = closest_value

if context:
    st.info(
        f"Extracted: {len(file_content)} characters. Ballpark estimated total tokens: {estimated_tokens - padding}"
    )

# Replace single submit button with two buttons side by side
col1, col2 = st.columns(2)
with col1:
    refine_button = st.button(
        "Refine Prompt", type="secondary", use_container_width=True
    )
with col2:
    submit_button = st.button("Submit to API", type="primary", use_container_width=True)

# A container at the top to display final answer
final_answer_placeholder = st.empty()


# Define a simple Gateway class to replace the imported one
class Gateway:
    def __init__(self, local_client=None, remote_client=None, callback=None):
        """Initialize the Gateway with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            callback: Optional callback function to receive message updates
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.callback = callback

    def __call__(self, task, context, max_rounds=None, doc_metadata=None):
        """Run the gateway protocol to refine and answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Not used in Gateway protocol but kept for API compatibility
            doc_metadata: Optional metadata about the document

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """
        # Join context sections
        context = "\n\n".join(context)

        # Add document metadata to context if provided
        if doc_metadata:
            context = f"{doc_metadata}\n\n{context}"

        # Initialize message histories and usage tracking
        worker_messages = [
            {
                "role": "system",
                "content": WORKER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": WORKER_USER_PROMPT.format(task=task),
            },
        ]

        from minions.usage import Usage

        remote_usage = Usage()
        local_usage = Usage()

        # Get worker's response (local model refines the prompt and context)
        if self.callback:
            self.callback("worker", None, is_final=False)

        worker_response, worker_usage, done_reason = self.local_client.chat(
            messages=worker_messages
        )
        local_usage += worker_usage

        worker_messages.append({"role": "assistant", "content": worker_response[0]})
        if self.callback:
            self.callback("worker", worker_messages[-1])

        # Prepare the refined input for the supervisor (remote model)
        refined_input = worker_response[0]

        # Initialize supervisor messages with the refined input
        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_PROMPT.format(
                    refined_input=refined_input, context=context
                ),
            }
        ]

        # Get supervisor's response (remote model generates the final answer)
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        supervisor_response, supervisor_usage = self.remote_client.chat(
            messages=supervisor_messages
        )
        remote_usage += supervisor_usage

        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )
        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # The supervisor's response is the final answer
        final_answer = supervisor_response[0]

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
        }


# Function to refine the prompt using the local model
def refine_prompt(query, context, doc_metadata):
    with st.status(f"Refining prompt...", expanded=True) as status:
        try:
            # Initialize clients if needed
            if (
                "local_client" not in st.session_state
                or "remote_client" not in st.session_state
                or "method" not in st.session_state
            ):
                st.write(f"Initializing clients...")
                initialize_clients(
                    local_model_name,
                    remote_model_name,
                    selected_provider,
                    local_temperature,
                    local_max_tokens,
                    remote_temperature,
                    remote_max_tokens,
                    provider_key,
                    num_ctx,
                )

            # Create worker messages
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": WORKER_USER_PROMPT.format(task=query),
                },
            ]

            # Call the worker model directly
            if st.session_state.callback:
                st.session_state.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason = (
                st.session_state.local_client.chat(messages=worker_messages)
            )

            worker_messages.append({"role": "assistant", "content": worker_response[0]})
            if st.session_state.callback:
                st.session_state.callback("worker", worker_messages[-1])

            status.update(label=f"Prompt refinement complete!", state="complete")

            # Return the refined prompt
            return worker_response[0], worker_usage
        except Exception as e:
            st.error(f"An error occurred during prompt refinement: {str(e)}")
            return None, None


# Function to submit the prompt to the remote API
def submit_to_api(query, context, doc_metadata):
    with st.status(f"Submitting to API...", expanded=True) as status:
        try:
            # Initialize clients if needed
            if (
                "local_client" not in st.session_state
                or "remote_client" not in st.session_state
                or "method" not in st.session_state
            ):
                st.write(f"Initializing clients...")
                initialize_clients(
                    local_model_name,
                    remote_model_name,
                    selected_provider,
                    local_temperature,
                    local_max_tokens,
                    remote_temperature,
                    remote_max_tokens,
                    provider_key,
                    num_ctx,
                )

            # Prepare the supervisor message with the query and context
            # please add a system messages asking the output to be in markdown format
            supervisor_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please provide your final response in markdown format.",
                },
                {
                    "role": "user",
                    "content": f"{query}\n\nAdditional Context (if provided):\n{context}",
                },
            ]

            # Call the supervisor model directly
            if st.session_state.callback:
                st.session_state.callback("supervisor", None, is_final=False)

            supervisor_response, supervisor_usage = st.session_state.remote_client.chat(
                messages=supervisor_messages
            )

            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            if st.session_state.callback:
                st.session_state.callback("supervisor", supervisor_messages[-1])

            status.update(label=f"API response received!", state="complete")

            # Import Usage class
            from minions.usage import Usage

            # Create output structure similar to Gateway protocol
            output = {
                "final_answer": supervisor_response[0],
                "supervisor_messages": supervisor_messages,
                "worker_messages": [],
                "remote_usage": supervisor_usage,
                "local_usage": Usage(
                    completion_tokens=0, prompt_tokens=0
                ),  # Zero tokens for direct API call
            }

            return output, 0, 0  # Return output and dummy timing values
        except Exception as e:
            st.error(f"An error occurred during API submission: {str(e)}")
            return None, 0, 0


# Handle the refine button click
if refine_button and user_query:
    # Call the refine_prompt function
    refined_prompt, worker_usage = refine_prompt(user_query, context, doc_metadata)

    if refined_prompt:
        # Store the refined prompt and worker usage in session state
        st.session_state["refined_prompt"] = refined_prompt
        st.session_state["worker_usage"] = worker_usage
        st.session_state["show_refinement_results"] = True

        # Display the refined prompt in a nice format
        st.markdown("---")
        tabs = st.tabs(["🚀 Original Query", "✨ Refined Prompt"])

        with tabs[0]:
            st.code(user_query, language="markdown")

        with tabs[1]:
            st.code(refined_prompt, language="markdown")

        # Add a note about the refined prompt being available in the text area
        st.success(
            "✅ Prompt refined successfully! The refined prompt has been placed in the text area above."
        )

        # Rerun the app to show the refined prompt in the text area
        st.rerun()
elif refine_button:
    st.error("Please enter a query before refining.")

# Display refinement results if available
if st.session_state.get("show_refinement_results", False) and st.session_state.get(
    "worker_usage"
):
    # Token usage section removed
    # Reset the flag after displaying
    st.session_state["show_refinement_results"] = False

# Handle the submit button click
if submit_button and user_query:
    output, setup_time, execution_time = submit_to_api(
        user_query, context, doc_metadata
    )

    if output:
        # Display final answer with enhanced styling
        st.markdown("---")  # Add a visual separator

        # Create tabs for Query, Refined Prompt (if available), and Response
        tabs = []
        tab_labels = ["🚀 Original Query"]

        # Add Refined Prompt tab if worker messages exist
        if output["worker_messages"] and len(output["worker_messages"]) > 2:
            tab_labels.append("✨ Refined Prompt")

        tab_labels.append("🎯 Final Response")

        tabs = st.tabs(tab_labels)

        # Original Query tab
        with tabs[0]:
            st.code(user_query, language="markdown")

        # Refined Prompt tab (if available)
        if output["worker_messages"] and len(output["worker_messages"]) > 2:
            with tabs[1]:
                refined_prompt = output["worker_messages"][2]["content"]
                st.code(refined_prompt, language="markdown")

        # Final Response tab
        with tabs[-1]:
            # Use expander for better readability of long responses
            st.markdown(output["final_answer"])

        # Add a success message
        st.success("✅ Response generated successfully!")

elif submit_button:
    st.error("Please enter a query before submitting.")
