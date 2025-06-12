import streamlit as st
import time

from minions.minion import Minion
from minions.minions import Minions
from minions.minions_mcp import SyncMinionsMCP, MCPConfigManager
from minions.minions_deep_research import DeepResearchMinions
from minions.utils.app_utils import render_deep_research_ui
from minions.utils.firecrawl_util import scrape_url
from minions.minion_cua import MinionCUA, KEYSTROKE_ALLOWED_APPS, SAFE_WEBSITE_DOMAINS
from minions.utils.inference_estimator import InferenceEstimator


# Instead of trying to import at startup, set voice_generation_available to None
# and only attempt import when voice generation is requested
voice_generation_available = None
voice_generator = None

from minions.clients import *

import os
import time
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import io
from pydantic import BaseModel
import json
from streamlit_theme import st_theme
from dotenv import load_dotenv
import base64
import requests


# Load environment variables from .env file
load_dotenv()

# Check if MLXLMClient, CartesiaMLXClient, TransformersClient, and CSM-MLX are available
mlx_available = "MLXLMClient" in globals()
cartesia_available = "CartesiaMLXClient" in globals()
transformers_available = "TransformersClient" in globals()
gemini_available = "GeminiClient" in globals()


# Log availability for debugging
print(f"MLXLMClient available: {mlx_available}")
print(f"CartesiaMLXClient available: {cartesia_available}")
print(f"TransformersClient available: {transformers_available}")
print(
    f"Voice generation available: {voice_generation_available if voice_generation_available is not None else 'Not checked yet'}"
)
print(f"GeminiClient available: {gemini_available}")


class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None


# Set custom sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 450px;
            max-width: 750px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

API_PRICES = {
    # OpenAI model pricing per 1M tokens
    "OpenAI": {
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gpt-4.5-preview": {"input": 75.00, "cached_input": 37.50, "output": 150.00},
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-pro": {"input": 150.00, "cached_input": 7.50, "output": 600.00},
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        "o3": {"input": 10.0, "cached_input": 2.50, "output": 40.00},
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    },
    # DeepSeek model pricing per 1M tokens
    "DeepSeek": {
        # Let's assume 1 dollar = 7.25 RMB and
        "deepseek-chat": {"input": 0.27, "cached_input": 0.07, "output": 1.10},
        "deepseek-reasoner": {"input": 0.27, "cached_input": 0.07, "output": 1.10},
    },
    "Gemini": {
        "gemini-2.5-pro-exp-06-05": {"input": 1.25, "cached_input": 0.075, "output": 10.0},
        "gemini-2.5-flash": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gemini-2.0-flash": {"input": 0.35, "cached_input": 0.175, "output": 1.05},
        "gemini-2.0-pro": {"input": 3.50, "cached_input": 1.75, "output": 10.50},
        "gemini-1.5-pro": {"input": 3.50, "cached_input": 1.75, "output": 10.50},
        "gemini-1.5-flash": {"input": 0.35, "cached_input": 0.175, "output": 1.05},
    },
    # LlamaAPI pricing (approximate - please update with actual pricing)
    "LlamaAPI": {
        "Llama-4-Maverick-17B-128E-Instruct-FP8": {
            "input": 0,
            "cached_input": 0,
            "output": 0,
        },
        "Llama-4-Scout-17B-16E-Instruct-FP8": {
            "input": 0,
            "cached_input": 0,
            "output": 0,
        },
        "Llama-3.3-70B-Instruct": {"input": 0, "cached_input": 0, "output": 0},
        "Llama-3.3-8B-Instruct": {"input": 0, "cached_input": 0, "output": 0},
    },
    # Sarvam AI pricing (approximate - please update with actual pricing)
    "Sarvam": {
        "sarvam-m": {"input": 0.50, "cached_input": 0.25, "output": 1.50},
        "sarvam-1b": {"input": 0.10, "cached_input": 0.05, "output": 0.30},
        "sarvam-3b": {"input": 0.30, "cached_input": 0.15, "output": 0.90},
    },
}

PROVIDER_TO_ENV_VAR_KEY = {
    "OpenAI": "OPENAI_API_KEY",
    "AzureOpenAI": "AZURE_OPENAI_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Together": "TOGETHER_API_KEY",
    "Perplexity": "PERPLEXITY_API_KEY",
    "Groq": "GROQ_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "Firecrawl": "FIRECRAWL_API_KEY",
    "SERP": "SERPAPI_API_KEY",
    "SambaNova": "SAMBANOVA_API_KEY",
    "Gemini": "GOOGLE_API_KEY",
    "HuggingFace": "HF_TOKEN",
    "LlamaAPI": "LLAMA_API_KEY",
    "Sarvam": "SARVAM_API_KEY",
}


# for Minions protocol
class JobOutput(BaseModel):
    answer: str | None
    explanation: str | None
    citation: str | None


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


# def extract_text_from_image(image_bytes):
#     """Extract text from an image file using pytesseract OCR."""
#     try:
#         import pytesseract

#         image = Image.open(io.BytesIO(image_bytes))
#         text = pytesseract.image_to_string(image)
#         return text
#     except Exception as e:
#         st.error(f"Error processing image: {str(e)}")
#         return None


def extract_text_from_image(path_to_file):
    try:
        # set up ollama client with model name="granite3.2-vision"
        client = OllamaClient(
            model_name="granite3.2-vision",
            use_async=False,
            num_ctx=131072,
        )
        responses, usage_total, done_reasons = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image:",
                    "images": [path_to_file],
                }
            ],
        )
        return responses[0]
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def jobs_callback(jobs):
    """Display a list of jobs with toggleable details."""
    total_jobs = len(jobs)
    successful_jobs = sum(1 for job in jobs if job.include)
    st.write(f"### Jobs ({successful_jobs}/{total_jobs} successful)")
    for job_idx, job in enumerate(jobs):
        icon = "✅" if job.include else "❌"
        with st.expander(
            f"{icon} Job {job_idx + 1} (Task: {job.manifest.task_id}, Chunk: {job.manifest.chunk_id})"
        ):
            st.write("**Task:**")
            st.write(job.manifest.task)
            st.write("**Chunk Preview:**")
            chunk_preview = (
                job.manifest.chunk[:100] + "..."
                if len(job.manifest.chunk) > 100
                else job.manifest.chunk
            )
            st.write(chunk_preview)
            if job.output.answer:
                st.write("**Answer:**")
                st.write(job.output.answer)
            if job.output.explanation:
                st.write("**Explanation:**")
                st.write(job.output.explanation)
            if job.output.citation:
                st.write("**Citation:**")
                st.write(job.output.citation)


THINKING_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2xhc3QzaHZyYWJ0M3czZXVjMGQ0YW50ZTBvcDdlNXVxNWhvZHdhOCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7bu3XilJ5BOiSGic/giphy.gif"
GRU_GIF = "https://media.giphy.com/media/ySMINwPzf50IM/giphy.gif?cid=790b7611vozglgf917p8ou0vjzydpgk9p8hpdwq9x95euttp&ep=v1_gifs_search&rid=giphy.gif&ct=g"
MINION_VIDEO = "https://www.youtube.com/embed/65BzWiQTkII?autoplay=1&mute=1"


def is_dark_mode():
    theme = st_theme()
    if theme and "base" in theme:
        if theme["base"] == "dark":
            return True
    return False


# Check theme setting
dark_mode = is_dark_mode()

# Choose image based on theme
if dark_mode:
    image_path = (
        "assets/minions_logo_no_background.png"  # Replace with your dark mode image
    )
else:
    image_path = "assets/minions_logo_light.png"  # Replace with your light mode image


# Display Minions logo at the top
st.image(image_path, use_container_width=True)

# add a horizontal line that is width of image
st.markdown("<hr style='width: 100%;'>", unsafe_allow_html=True)


def message_callback(role, message, is_final=True):
    """Show messages for both Minion and Minions protocols,
    labeling the local vs remote model clearly."""
    
    # Initialize placeholder_messages in session state if not present
    if "placeholder_messages" not in st.session_state:
        st.session_state.placeholder_messages = {}
    
    # Map supervisor -> Remote, worker -> Local
    chat_role = "Remote" if role == "supervisor" else "Local"

    if role == "supervisor":
        chat_role = "Remote"
        path = "assets/gru.jpg"
        # path = GRU_GIF
    else:
        chat_role = "Local"
        path = "assets/minion.png"
        # path = MINION_GIF

    # If we are not final, handle intermediate content or show working state
    if not is_final:
        # Clear any existing placeholder for this role to prevent accumulation
        if role in st.session_state.placeholder_messages:
            try:
                st.session_state.placeholder_messages[role].empty()
            except Exception as e:
                print(f"Warning: Could not clear existing placeholder for {role}: {e}")
        
        # Create a placeholder container and store it for later update.
        placeholder = st.empty()
        
        # Check if we have actual content to show or just need to show working state
        has_content = message is not None and message != ""
        
        if has_content:
            # Show intermediate content
            with placeholder.chat_message(chat_role, avatar=path):
                # Generate voice if enabled and it's a worker/local message
                if (
                    st.session_state.get("voice_generation_enabled", False)
                    and role == "worker"
                    and "voice_generator" in st.session_state
                ):
                    # For text messages, generate audio
                    if isinstance(message, str):
                        # Limit text length for voice generation
                        voice_text = (
                            message[:500] + "..." if len(message) > 500 else message
                        )
                        audio_base64 = st.session_state.voice_generator.generate_audio(
                            voice_text
                        )
                        if audio_base64:
                            st.markdown(
                                st.session_state.voice_generator.get_audio_html(
                                    audio_base64
                                ),
                                unsafe_allow_html=True,
                            )
                    elif isinstance(message, dict):
                        if "content" in message and isinstance(message["content"], str):
                            voice_text = (
                                message["content"][:500] + "..."
                                if len(message["content"]) > 500
                                else message["content"]
                            )
                            audio_base64 = st.session_state.voice_generator.generate_audio(
                                voice_text
                            )
                            if audio_base64:
                                st.markdown(
                                    st.session_state.voice_generator.get_audio_html(
                                        audio_base64
                                    ),
                                    unsafe_allow_html=True,
                                )

                # Display the intermediate content
                if role == "worker" and isinstance(message, list):
                    # For Minions protocol, messages are a list of jobs (intermediate results)
                    st.markdown("#### Intermediate results from minions...")
                    tasks = {}
                    for job in message:
                        task_id = job.manifest.task_id
                        if task_id not in tasks:
                            tasks[task_id] = {"task": job.manifest.task, "jobs": []}
                        tasks[task_id]["jobs"].append(job)

                    for task_id, task_info in tasks.items():
                        task_info["jobs"] = sorted(
                            task_info["jobs"], key=lambda x: x.manifest.job_id
                        )
                        include_jobs = [
                            job
                            for job in task_info["jobs"]
                            if job.output.answer
                            and job.output.answer.lower().strip() != "none"
                        ]

                        st.markdown(
                            f"_Processing: {len(task_info['jobs']) - len(include_jobs)} chunks pending, {len(include_jobs)} completed_"
                        )
                        # Show a sample of completed jobs
                        for job in include_jobs[:3]:  # Show first 3 completed jobs
                            st.markdown(
                                f"**✅ Job {job.manifest.job_id + 1} (Chunk {job.manifest.chunk_id + 1})**"
                            )
                            answer = job.output.answer.replace("$", "\\$")
                            st.markdown(f"Answer: {answer}")

                elif isinstance(message, dict):
                    if "content" in message and isinstance(message["content"], (dict, str)):
                        try:
                            # Try to parse as JSON if it's a string
                            content = (
                                message["content"]
                                if isinstance(message["content"], dict)
                                else json.loads(message["content"])
                            )
                            st.json(content)
                        except json.JSONDecodeError:
                            st.write(message["content"])
                    else:
                        st.write(message)
                elif isinstance(message, str):
                    message = message.replace("$", "\\$")
                    st.markdown(message)
                else:
                    st.write(str(message))
        else:
            # Show working state when no content
            with placeholder.chat_message(chat_role, avatar=path):
                st.markdown("**Working...**")
                if role == "supervisor":
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center;">
                            <img src="{GRU_GIF}" width="200">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    video_html = f"""
                        <style>
                        .video-container {{
                            position: relative;
                            padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
                            height: 0;
                            overflow: hidden;
                            max-width: 100%;
                        }}
                        .video-container iframe {{
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                        }}
                        </style>

                        <div class="video-container">
                            <iframe src="{MINION_VIDEO}"
                            frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
                        </div>
                    """
                    st.markdown(video_html, unsafe_allow_html=True)
        
        st.session_state.placeholder_messages[role] = placeholder
    else:
        # Handle final message - clear placeholder for this role
        if role in st.session_state.placeholder_messages:
            try:
                st.session_state.placeholder_messages[role].empty()
                del st.session_state.placeholder_messages[role]
            except Exception as e:
                # Handle edge case where placeholder might be invalid
                print(f"Warning: Could not clear placeholder for {role}: {e}")
                # Remove from dict anyway
                if role in st.session_state.placeholder_messages:
                    del st.session_state.placeholder_messages[role]
        with st.chat_message(chat_role, avatar=path):
            # Generate voice if enabled and it's a worker/local message
            if (
                st.session_state.get("voice_generation_enabled", False)
                and role == "worker"
                and "voice_generator" in st.session_state
            ):
                # For text messages, generate audio
                if isinstance(message, str):
                    # Limit text length for voice generation
                    voice_text = (
                        message[:500] + "..." if len(message) > 500 else message
                    )
                    audio_base64 = st.session_state.voice_generator.generate_audio(
                        voice_text
                    )
                    if audio_base64:
                        st.markdown(
                            st.session_state.voice_generator.get_audio_html(
                                audio_base64
                            ),
                            unsafe_allow_html=True,
                        )
                elif isinstance(message, dict):
                    if "content" in message and isinstance(message["content"], str):
                        voice_text = (
                            message["content"][:500] + "..."
                            if len(message["content"]) > 500
                            else message["content"]
                        )
                        audio_base64 = st.session_state.voice_generator.generate_audio(
                            voice_text
                        )
                        if audio_base64:
                            st.markdown(
                                st.session_state.voice_generator.get_audio_html(
                                    audio_base64
                                ),
                                unsafe_allow_html=True,
                            )

            if role == "worker" and isinstance(message, list):
                # For Minions protocol, messages are a list of jobs
                st.markdown("#### Here are the outputs from all the minions!")
                tasks = {}
                for job in message:
                    task_id = job.manifest.task_id
                    if task_id not in tasks:
                        tasks[task_id] = {"task": job.manifest.task, "jobs": []}
                    tasks[task_id]["jobs"].append(job)

                for task_id, task_info in tasks.items():
                    # first srt task_info[jobs] by job_id
                    task_info["jobs"] = sorted(
                        task_info["jobs"], key=lambda x: x.manifest.job_id
                    )
                    include_jobs = [
                        job
                        for job in task_info["jobs"]
                        if job.output.answer
                        and job.output.answer.lower().strip() != "none"
                    ]

                    st.markdown(
                        f"_Note: {len(task_info['jobs']) - len(include_jobs)} jobs did not have relevant information._"
                    )
                    st.markdown(f"**Jobs with relevant information:**")
                    # print all the relevant information
                    for job in include_jobs:
                        st.markdown(
                            f"**✅ Job {job.manifest.job_id + 1} (Chunk {job.manifest.chunk_id + 1})**"
                        )
                        answer = job.output.answer.replace("$", "\\$")
                        st.markdown(f"Answer: {answer}")

            elif isinstance(message, dict):
                if "content" in message and isinstance(message["content"], (dict, str)):
                    try:
                        # Try to parse as JSON if it's a string
                        content = (
                            message["content"]
                            if isinstance(message["content"], dict)
                            else json.loads(message["content"])
                        )
                        st.json(content)

                    except json.JSONDecodeError:
                        st.write(message["content"])
                else:
                    st.write(message)
            else:
                message = message.replace("$", "\\$")
                st.markdown(message)


def initialize_clients(
    local_model_name,
    remote_model_name,
    provider,
    local_provider,
    protocol,
    local_temperature,
    local_max_tokens,
    remote_temperature,
    remote_max_tokens,
    api_key,
    num_ctx=4096,
    mcp_server_name=None,
    reasoning_effort="medium",
    multi_turn_mode=False,
    max_history_turns=0,
    context_description=None,
):
    """Initialize the local and remote clients outside of the run_protocol function."""
    # Store model parameters in session state for potential reinitialization
    st.session_state.local_model_name = local_model_name
    st.session_state.remote_model_name = remote_model_name
    st.session_state.local_temperature = local_temperature
    st.session_state.local_max_tokens = local_max_tokens
    st.session_state.remote_temperature = remote_temperature
    st.session_state.remote_max_tokens = remote_max_tokens
    st.session_state.provider = provider
    st.session_state.local_provider = local_provider
    st.session_state.api_key = api_key
    st.session_state.mcp_server_name = mcp_server_name
    st.session_state.multi_turn_mode = multi_turn_mode
    st.session_state.max_history_turns = max_history_turns

    # Store thinking_budget if provider is Gemini
    if provider == "Gemini" and "thinking_budget" in st.session_state:
        st.session_state.thinking_budget = st.session_state.thinking_budget

    # Store context description in session state
    if context_description:
        st.session_state.context_description = context_description

    # For Minions we want asynchronous local chunk processing:
    if protocol in ["Minions", "Minions-MCP"]:
        use_async = True
        # For Minions, we use a fixed context size since it processes chunks
        minions_ctx = 4096

        # Use appropriate client based on local provider
        if local_provider == "MLX":
            st.session_state.local_client = MLXLMClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
            )
        elif local_provider == "Cartesia-MLX":
            st.session_state.local_client = CartesiaMLXClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
            )

        else:  # Ollama
            st.session_state.local_client = OllamaClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
                num_ctx=minions_ctx,
                structured_output_schema=StructuredLocalOutput,
                use_async=use_async,
            )
    else:
        use_async = False

        # Use appropriate client based on local provider
        if local_provider == "MLX":
            st.session_state.local_client = MLXLMClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
            )
        elif local_provider == "Cartesia-MLX":
            st.session_state.local_client = CartesiaMLXClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
            )
        elif local_provider == "Transformers":
            hf_token = os.getenv("HF_TOKEN")
            st.session_state.local_client = TransformersClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
                hf_token=hf_token,
                do_sample=(local_temperature > 0),
            )
        elif local_provider == "Modular":
            # Get Modular configuration from session state
            modular_model_name = st.session_state.get("modular_model_name", local_model_name)
            modular_weights_path = st.session_state.get("modular_weights_path", "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/smollm2-1.7b-instruct-q4_k_m.gguf")
            modular_verbose = st.session_state.get("modular_verbose", False)
            
            from minions.clients.modular import ModularClient
            st.session_state.local_client = ModularClient(
                model_name=modular_model_name,
                weights_path=modular_weights_path,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
                verbose=modular_verbose,
            )
        else:  # Ollama
            st.session_state.local_client = OllamaClient(
                model_name=local_model_name,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
                num_ctx=num_ctx,
                structured_output_schema=None,
                use_async=use_async,
            )

        # Calibrate the inference estimator with the local client
        try:
            st.session_state.inference_estimator = InferenceEstimator(local_model_name)
            # st.session_state.inference_estimator.calibrate(
            #     st.session_state.local_client
            # )
        except Exception as e:
            # Log but don't crash if calibration fails
            print(f"Calibration failed: {str(e)}")

    if provider == "OpenAI":
        # Add web search tool if responses API is enabled
        tools = None
        if use_responses_api:
            tools = [{"type": "web_search_preview"}]

        st.session_state.remote_client = OpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
            use_responses_api=use_responses_api,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
    elif provider == "AzureOpenAI":
        # Get Azure-specific parameters from environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        azure_api_key = api_key if api_key else os.getenv("AZURE_OPENAI_API_KEY")

        # Show warning if endpoint is not set
        if not azure_endpoint:
            st.warning(
                "Azure OpenAI endpoint not set. Please set the AZURE_OPENAI_ENDPOINT environment variable."
            )
            st.info(
                "You can run the setup_azure_openai.sh script to configure Azure OpenAI settings."
            )
        else:
            st.success(f"Using Azure OpenAI endpoint: {azure_endpoint}")

        st.session_state.remote_client = AzureOpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )
    elif provider == "OpenRouter":
        st.session_state.remote_client = OpenRouterClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Anthropic":
        # Get web search settings directly from session state
        use_web_search = st.session_state.get("use_web_search", False)
        include_search_queries = st.session_state.get("include_search_queries", False)
        max_web_search_uses = st.session_state.get("max_web_search_uses", 5)
        use_caching = st.session_state.get("use_caching", False)
        use_code_interpreter = st.session_state.get("use_code_interpreter", False)

        st.session_state.remote_client = AnthropicClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
            use_web_search=use_web_search,
            include_search_queries=include_search_queries,
            use_caching=use_caching,
            use_code_interpreter=use_code_interpreter,
        )

        # Set max_web_search_uses if web search is enabled
        if use_web_search:
            st.session_state.remote_client.max_web_search_uses = max_web_search_uses
    elif provider == "Together":
        st.session_state.remote_client = TogetherClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Perplexity":
        st.session_state.remote_client = PerplexityAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Groq":
        st.session_state.remote_client = GroqClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "DeepSeek":
        st.session_state.remote_client = DeepSeekClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "SambaNova":
        st.session_state.remote_client = SambanovaClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Gemini":
        st.session_state.remote_client = GeminiClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
            thinking_budget=st.session_state.get("thinking_budget", 0),
        )
    elif provider == "LlamaAPI":
        st.session_state.remote_client = LlamaApiClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Sarvam":
        st.session_state.remote_client = SarvamClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )
    elif provider == "Secure":
        # Get secure endpoint URL from session state or environment
        secure_endpoint_url = st.session_state.get("secure_endpoint_url") or os.getenv(
            "SECURE_ENDPOINT_URL"
        )
        verify_attestation = st.session_state.get("verify_attestation", True)
        session_timeout = st.session_state.get("session_timeout", 3600)

        if not secure_endpoint_url:
            st.error(
                "Secure endpoint URL is required. Please set it in the configuration."
            )
            secure_endpoint_url = "https://localhost:8000"  # Default fallback

        try:
            from minions.clients.secure import SecureClient

            st.session_state.remote_client = SecureClient(
                endpoint_url=secure_endpoint_url,
                verify_attestation=verify_attestation,
                session_timeout=session_timeout,
            )
        except ImportError:
            st.error(
                "SecureClient is not available. Please ensure the secure module is properly installed."
            )
            # Fallback to OpenAI client
            remote_model_name = "gpt-4o"
            st.session_state.remote_client = OpenAIClient(
                model_name=remote_model_name,
                temperature=remote_temperature,
                max_tokens=int(remote_max_tokens),
                api_key=api_key,
            )
    else:  # OpenAI
        st.session_state.remote_client = OpenAIClient(
            model_name=remote_model_name,
            temperature=remote_temperature,
            max_tokens=int(remote_max_tokens),
            api_key=api_key,
        )

    if protocol == "Minions":
        st.session_state.method = Minions(
            st.session_state.local_client,
            st.session_state.remote_client,
            callback=message_callback,
        )

        # Initialize chunking_fn if not already set
        if "chunk_fn" not in st.session_state:
            st.session_state.chunk_fn = "chunk_by_section"
    elif protocol == "Minions-MCP":
        st.session_state.method = SyncMinionsMCP(
            local_client=st.session_state.local_client,
            remote_client=st.session_state.remote_client,
            mcp_server_name=mcp_server_name,
            callback=message_callback,
        )
    elif protocol == "DeepResearch":
        st.session_state.method = DeepResearchMinions(
            local_client=st.session_state.local_client,
            remote_client=st.session_state.remote_client,
            callback=message_callback,
            max_rounds=3,
            max_sources_per_round=5,
        )
    elif protocol == "Minion-CUA":
        print(protocol)
        print("Initializing Minion-CUA protocol...")

        # Import the custom CUA initial prompt
        from minions.prompts.minion_cua import SUPERVISOR_CUA_INITIAL_PROMPT

        # Create a custom initialization message to pass to remote models
        init_message = {
            "role": "system",
            "content": "This assistant has direct control over the user's computer. It can perform real physical automation on macOS, including opening applications, typing text, clicking elements, using keyboard shortcuts, and opening URLs.",
        }

        # Initialize with the specialized CUA class
        st.session_state.method = MinionCUA(
            st.session_state.local_client,
            st.session_state.remote_client,
            callback=message_callback,
        )

        # Test if the class is correctly initialized
        print(f"Method type: {type(st.session_state.method).__name__}")
    else:  # Minion protocol
        st.session_state.method = Minion(
            st.session_state.local_client,
            st.session_state.remote_client,
            callback=message_callback,
            is_multi_turn=multi_turn_mode,
            max_history_turns=max_history_turns,
        )

    # Get reasoning_effort from the widget value directly
    if "reasoning_effort" in st.session_state:
        reasoning_effort = st.session_state.reasoning_effort
    else:
        reasoning_effort = "medium"  # Default if not set

    (
        st.session_state.local_client,
        st.session_state.remote_client,
        st.session_state.method,
    ) = (
        st.session_state.local_client,
        st.session_state.remote_client,
        st.session_state.method,
    )

    return (
        st.session_state.local_client,
        st.session_state.remote_client,
        st.session_state.method,
    )


def run_protocol(
    task,
    context,
    doc_metadata,
    status,
    protocol,
    local_provider,
    images=None,
):
    """Run the protocol with pre-initialized clients."""
    setup_start_time = time.time()

    # Add context_description to doc_metadata if it exists
    if (
        "context_description" in st.session_state
        and st.session_state.context_description
    ):
        if doc_metadata:
            doc_metadata = f"{doc_metadata} (Context Description: {st.session_state.context_description})"
        else:
            doc_metadata = (
                f"Context Description: {st.session_state.context_description}"
            )

    with status.container():
        messages_container = st.container()
        st.markdown(f"**Query:** {task}")

        # If context size has changed, we need to update the local client's num_ctx
        # But only for Minion protocol, not Minions (which processes chunks)
        if (
            "local_client" in st.session_state
            and hasattr(st.session_state.local_client, "num_ctx")
            and protocol == "Minion"
            and st.session_state.current_protocol == "Minion"
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

                # According to Ollama documentation, num_ctx needs to be set during initialization
                # So we need to reinitialize the local client with the new num_ctx
                if (
                    "local_model_name" in st.session_state
                    and "local_temperature" in st.session_state
                    and "local_max_tokens" in st.session_state
                    and "api_key" in st.session_state
                ):

                    # Reinitialize the local client with the new num_ctx
                    if local_provider == "Ollama":
                        st.session_state.local_client = OllamaClient(
                            model_name=st.session_state.local_model_name,
                            temperature=st.session_state.local_temperature,
                            max_tokens=int(st.session_state.local_max_tokens),
                            num_ctx=closest_value,
                            structured_output_schema=None,  # Minion protocol doesn't use structured output
                            use_async=False,  # Minion protocol doesn't use async
                        )
                    elif local_provider == "Transformers":
                        hf_token = os.getenv("HF_TOKEN")
                        st.session_state.local_client = TransformersClient(
                            model_name=st.session_state.local_model_name,
                            temperature=st.session_state.local_temperature,
                            max_tokens=int(st.session_state.local_max_tokens),
                            hf_token=hf_token,
                            do_sample=(st.session_state.local_temperature > 0),
                        )
                    else:
                        st.session_state.local_client = MLXLMClient(
                            model_name=st.session_state.local_model_name,
                            temperature=st.session_state.local_temperature,
                            max_tokens=int(st.session_state.local_max_tokens),
                        )

                    # Reinitialize the method with the new local client
                    st.session_state.method = Minion(
                        st.session_state.local_client,
                        st.session_state.remote_client,
                        callback=message_callback,
                        multi_turn_mode=st.session_state.get("multi_turn_mode", False),
                        max_history_turns=st.session_state.get("max_history_turns", 0),
                    )

        if "inference_estimator" in st.session_state:
            try:
                tokens_per_second, eta = st.session_state.inference_estimator.estimate(
                    estimated_tokens
                )
                if eta > 0:
                    st.write(
                        f"Estimated completion time: {eta:.2f} seconds ({tokens_per_second:.1f} tokens/sec)"
                    )
            except Exception as e:
                st.write(f"Could not estimate completion time: {str(e)}")

        setup_time = time.time() - setup_start_time
        st.write("Solving task...")
        execution_start_time = time.time()

        # Call the appropriate protocol with the correct parameters
        output = None  # Initialize output to avoid reference errors

        # Add timing wrappers to the clients to track time spent in each
        # Create timing wrappers for the clients
        local_time_spent = 0
        remote_time_spent = 0

        # Store original chat methods
        original_local_chat = st.session_state.local_client.chat
        original_remote_chat = st.session_state.remote_client.chat

        # Create timing wrapper for local client
        def timed_local_chat(*args, **kwargs):
            nonlocal local_time_spent
            start_time = time.time()
            result = original_local_chat(*args, **kwargs)
            local_time_spent += time.time() - start_time
            return result

        # Create timing wrapper for remote client
        def timed_remote_chat(*args, **kwargs):
            nonlocal remote_time_spent
            start_time = time.time()
            result = original_remote_chat(*args, **kwargs)
            remote_time_spent += time.time() - start_time
            return result

        # Replace the chat methods with the timed versions
        st.session_state.local_client.chat = timed_local_chat
        st.session_state.remote_client.chat = timed_remote_chat

        # Pass is_privacy parameter when using Minion protocol
        if protocol == "Minion":
            output = st.session_state.method(
                task=task,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=2,
                is_privacy=privacy_mode,  # Pass the privacy mode setting
                images=images,
            )
        elif protocol == "Minions":
            use_retrieval = None
            if use_bm25:
                use_retrieval = "bm25"
            elif use_retrieval == "embedding":
                use_retrieval = "multimodal-embedding"

            # Show info about the selected chunking method
            chunk_methods = {
                "chunk_by_section": "Section",
                "chunk_by_page": "Page",
                "chunk_by_paragraph": "Paragraph",
                "chunk_by_code": "Code",
                "chunk_by_function_and_class": "Function and Class",
            }
            chunk_method_display = chunk_methods.get(
                st.session_state.chunk_fn, "Custom"
            )

            st.info(
                f"📄 Using '{chunk_method_display}' chunking method to process your document"
            )

            output = st.session_state.method(
                task=task,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=5,
                max_jobs_per_round=max_jobs_per_round,
                use_retrieval=use_retrieval,
                chunk_fn=st.session_state.chunk_fn,  # Pass the selected chunking function
            )
        elif protocol == "DeepResearch":
            output = st.session_state.method(
                query=task,
                firecrawl_api_key=st.session_state.get("firecrawl_api_key"),
                serpapi_key=st.session_state.get("serpapi_api_key"),
            )
        elif protocol == "Minion-CUA":
            # For CUA, let's be clear about automation capabilities
            st.info("💡 Using Computer User Automation to physically control your Mac")
            output = st.session_state.method(
                task=task,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=5,
                images=images,
            )

        else:
            output = st.session_state.method(
                task=task,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=5,
                use_bm25=False,
            )

        execution_time = time.time() - execution_start_time

        # Restore original chat methods
        st.session_state.local_client.chat = original_local_chat
        st.session_state.remote_client.chat = original_remote_chat

        # Add timing information to output
        output["local_time"] = local_time_spent
        output["remote_time"] = remote_time_spent
        output["other_time"] = execution_time - (local_time_spent + remote_time_spent)

    return output, setup_time, execution_time


def validate_sambanova_key(api_key):
    try:
        client = SambanovaClient(
            model_name="Meta-Llama-3.1-8B-Instruct",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_gemini_key(api_key):
    try:
        client = GeminiClient(
            model_name="gemini-2.0-flash",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


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


def validate_anthropic_key(api_key):
    try:
        client = AnthropicClient(
            model_name="claude-3-5-haiku-latest",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
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


def validate_perplexity_key(api_key):
    try:
        client = PerplexityAIClient(
            model_name="sonar-pro", api_key=api_key, temperature=0.0, max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_openrouter_key(api_key):
    try:
        client = OpenRouterClient(
            model_name="anthropic/claude-3.5-sonnet",  # Use a common model for testing
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_groq_key(api_key):
    try:
        client = GroqClient(
            model_name="llama-3.3-70b-versatile",  # Use a common model for testing
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_deepseek_key(api_key):
    try:
        client = DeepSeekClient(
            model_name="deepseek-chat", api_key=api_key, temperature=0.0, max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_azure_openai_key(api_key):
    """Validate Azure OpenAI API key by checking if it's not empty."""
    if not api_key:
        return False, "API key is empty"

    # Azure OpenAI keys are typically 32 characters long
    if len(api_key) < 10:  # Simple length check
        return False, "API key is too short"

    # We can't make a test call here without the endpoint
    # So we just do basic validation
    return True, "API key format is valid"


def validate_huggingface_token(hf_token):
    """Validate HuggingFace API token by checking if it's not empty."""
    if not hf_token:
        return False, "HuggingFace token is empty"

    # HuggingFace tokens are typically at least 30 characters long
    if len(hf_token) < 10:  # Simple length check
        return False, "HuggingFace token is too short"

    # We can't easily make a test call here without a specific model
    # So we just do basic validation
    return True, "HuggingFace token format is valid"


def validate_llama_api_key(api_key):
    try:
        client = LlamaApiClient(
            model_name="Llama-3.3-8B-Instruct",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_sarvam_key(api_key):
    try:
        client = SarvamClient(
            model_name="sarvam-m",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1,
        )
        messages = [{"role": "user", "content": "Say yes"}]
        client.chat(messages)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_secure_endpoint(endpoint_url):
    """Validate secure endpoint by checking if it uses HTTPS and is reachable."""
    try:
        import requests
        from urllib.parse import urlparse

        if not endpoint_url:
            return False, "Endpoint URL is required"

        # Validate HTTPS protocol
        parsed_url = urlparse(endpoint_url)
        if parsed_url.scheme != "https":
            return (
                False,
                f"Secure endpoint must use HTTPS protocol. Got: {parsed_url.scheme}://{parsed_url.netloc}",
            )

        if not parsed_url.netloc:
            return False, "Invalid endpoint URL format"

        # Check if the endpoint is reachable
        response = requests.get(f"{endpoint_url}/health", timeout=10)
        if response.status_code == 200:
            return True, "Secure endpoint is reachable and uses HTTPS"
        else:
            return False, f"Endpoint returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to secure endpoint"
    except requests.exceptions.Timeout:
        return False, "Connection to secure endpoint timed out"
    except Exception as e:
        return False, f"Error validating secure endpoint: {str(e)}"


# validate


# ---------------------------
#  Sidebar for LLM settings
# ---------------------------
with st.sidebar:
    st.subheader("LLM Provider Settings")

    # Remote provider selection
    provider_col, key_col = st.columns([1, 2])
    with provider_col:
        # List of remote providers
        providers = [
            "OpenAI",
            "AzureOpenAI",
            "OpenRouter",
            "Anthropic",
            "Together",
            "Perplexity",
            "Groq",
            "DeepSeek",
            "SambaNova",
            "Gemini",
            "LlamaAPI",
            "Sarvam",
            "Secure",
        ]
        selected_provider = st.selectbox(
            "Select Remote Provider",
            options=providers,
            index=0,
        )  # Set OpenAI as default (index 0)

    # API key handling for remote provider
    env_var_name = f"{selected_provider.upper()}_API_KEY"
    env_key = os.getenv(env_var_name)
    with key_col:
        # For secure client, we don't need an API key
        if selected_provider == "Secure":
            st.info("🔒 Secure client uses endpoint-based authentication")
            user_key = ""
        else:
            user_key = st.text_input(
                f"{selected_provider} API Key (optional if set in environment)",
                type="password",
                value="",
                key=f"{selected_provider}_key",
            )
    api_key = user_key if user_key else env_key

    # Validate API key or endpoint
    if selected_provider == "Secure":
        # For secure client, validate the endpoint instead of API key
        secure_endpoint_url = st.session_state.get("secure_endpoint_url") or os.getenv(
            "SECURE_ENDPOINT_URL"
        )
        if secure_endpoint_url:
            is_valid, msg = validate_secure_endpoint(secure_endpoint_url)
            provider_key = secure_endpoint_url  # Use endpoint URL as the "key"
        else:
            st.error(
                "**✗ Missing secure endpoint URL.** Use https://minions-tee.eastus2.cloudapp.azure.com:443 if you don't have a custom endpoint."
            )
            provider_key = None
    elif api_key:
        if selected_provider == "OpenAI":
            is_valid, msg = validate_openai_key(api_key)
        elif selected_provider == "AzureOpenAI":
            is_valid, msg = validate_azure_openai_key(api_key)
        elif selected_provider == "OpenRouter":
            is_valid, msg = validate_openrouter_key(api_key)
        elif selected_provider == "Anthropic":
            is_valid, msg = validate_anthropic_key(api_key)
        elif selected_provider == "Together":
            is_valid, msg = validate_together_key(api_key)
        elif selected_provider == "Perplexity":
            is_valid, msg = validate_perplexity_key(api_key)
        elif selected_provider == "Groq":
            is_valid, msg = validate_groq_key(api_key)
        elif selected_provider == "DeepSeek":
            is_valid, msg = validate_deepseek_key(api_key)
        elif selected_provider == "SambaNova":
            is_valid, msg = validate_sambanova_key(api_key)
        elif selected_provider == "Gemini":
            is_valid, msg = validate_gemini_key(api_key)
        elif selected_provider == "LlamaAPI":
            is_valid, msg = validate_llama_api_key(api_key)
        elif selected_provider == "Sarvam":
            is_valid, msg = validate_sarvam_key(api_key)
        elif selected_provider == "Secure":
            # For secure client, validate the endpoint instead of API key
            secure_endpoint_url = st.session_state.get(
                "secure_endpoint_url"
            ) or os.getenv("SECURE_ENDPOINT_URL")
            if secure_endpoint_url:
                is_valid, msg = validate_secure_endpoint(secure_endpoint_url)
                if is_valid:
                    st.success("**✓ Secure endpoint is reachable.** You're good to go!")
                    provider_key = secure_endpoint_url  # Use endpoint URL as the "key"
                else:
                    st.error(f"**✗ Secure endpoint validation failed.** {msg}")
                    provider_key = None
            else:
                st.error(
                    "**✗ Missing secure endpoint URL.** Please configure the endpoint URL above."
                )
                provider_key = None
        else:
            raise ValueError(f"Invalid provider: {selected_provider}")

        if is_valid:
            st.success("**✓ Valid API key.** You're good to go!")
            provider_key = api_key
        else:
            st.error(f"**✗ Invalid API key.** {msg}")
            provider_key = None
    else:
        # Only show missing API key error for providers that actually need API keys
        if selected_provider not in ["Secure"]:
            st.error(
                f"**✗ Missing API key.** Input your key above or set the environment variable with `export {PROVIDER_TO_ENV_VAR_KEY[selected_provider]}=<your-api-key>`"
            )
            provider_key = None
        else:
            # For providers that don't need API keys, this shouldn't happen, but handle gracefully
            provider_key = None

    # Add a toggle for OpenAI Responses API with web search when OpenAI is selected
    if selected_provider == "OpenAI":
        use_responses_api = st.toggle(
            "Enable Responses API",
            value=False,
            help="When enabled, uses OpenAI's Responses API with web search capability. Only works with OpenAI provider.",
        )
    else:
        use_responses_api = False

    # Add a toggle for Anthropic web search when Anthropic is selected
    if selected_provider == "Anthropic":
        use_web_search = st.toggle(
            "Enable Web Search",
            value=st.session_state.get("use_web_search", False),
            key="use_web_search",
            help="When enabled, allows Claude to search the web for up-to-date information. Only works with Anthropic provider.",
        )

        # Remove redundant assignment - the toggle widget already updates session state
        # st.session_state.use_web_search = use_web_search

        # Show informational text about web search requirements
        if use_web_search:
            current_model = st.session_state.get(
                "remote_model", ""
            ) or st.session_state.get("current_remote_model", "")

            if "claude-3-7" not in str(current_model).lower():
                st.info(
                    "💡 Web search works best with Claude 3.7 Sonnet or newer models. "
                    "Consider selecting 'claude-3-7-sonnet-latest' from the Model dropdown below. "
                    "Responses will include citations to web sources."
                )
            else:
                st.info(
                    "💡 Web search is enabled. Claude will search the web for up-to-date information "
                    "and include citations to sources in its responses."
                )

            # Option to include search queries in the response
            include_search_queries = st.toggle(
                "Show Search Queries",
                value=st.session_state.get("include_search_queries", False),
                key="include_search_queries",
                help="When enabled, includes the search queries Claude used in the response.",
            )

            # Maximum web search uses
            max_web_search_uses = st.slider(
                "Max Web Searches",
                min_value=1,
                max_value=10,
                value=st.session_state.get("max_web_search_uses", 5),
                key="max_web_search_uses",
                help="Maximum number of web searches Claude can perform in a single request.",
            )

            # Remove redundant assignments - the widgets already update session state
            # st.session_state.include_search_queries = include_search_queries
            # st.session_state.max_web_search_uses = max_web_search_uses
        else:
            # This can be kept as it's not tied to a widget
            st.session_state.include_search_queries = False

        # Add caching and code interpreter options
        use_caching = st.toggle(
            "Enable Caching",
            value=st.session_state.get("use_caching", False),
            key="use_caching",
            help="When enabled, uses Anthropic's caching feature to reduce costs for repeated content. Useful for long documents or repeated queries.",
        )

        use_code_interpreter = st.toggle(
            "Enable Code Interpreter",
            value=st.session_state.get("use_code_interpreter", False),
            key="use_code_interpreter",
            help="When enabled, allows Claude to execute Python code to solve problems, analyze data, and generate visualizations.",
        )

        if use_code_interpreter:
            st.info(
                "💡 Code interpreter is enabled. Claude can now execute Python code to solve complex problems, "
                "perform calculations, analyze data, and create visualizations."
            )
    # Add thinking_budget option for Gemini provider
    elif selected_provider == "Gemini":
        thinking_budget = st.slider(
            "Thinking Budget",
            min_value=0,
            max_value=24576,
            value=st.session_state.get("thinking_budget", 0),
            step=1024,
            key="thinking_budget",
            help="Number of tokens used for model thinking before generating a response. 0 disables thinking. Higher values can improve response quality but increase token usage and cost.",
        )

    # Add secure client configuration
    elif selected_provider == "Secure":
        st.subheader("🔒 Secure Client Configuration")

        # Endpoint URL configuration
        secure_endpoint_url = st.text_input(
            "Secure Endpoint URL",
            value=st.session_state.get("secure_endpoint_url", "")
            or os.getenv("SECURE_ENDPOINT_URL", ""),
            key="secure_endpoint_url",
            help="URL of the secure endpoint (e.g., https://minions-tee.eastus2.cloudapp.azure.com:443)",
            placeholder="https://minions-tee.eastus2.cloudapp.azure.com:443",
        )

        # Attestation verification toggle
        verify_attestation = st.toggle(
            "Verify Attestation",
            value=st.session_state.get("verify_attestation", True),
            key="verify_attestation",
            help="When enabled, verifies the endpoint's attestation for enhanced security. Disable only for testing.",
        )

        # Session timeout configuration
        session_timeout = st.slider(
            "Session Timeout (seconds)",
            min_value=300,  # 5 minutes
            max_value=7200,  # 2 hours
            value=st.session_state.get("session_timeout", 3600),  # 1 hour default
            step=300,
            key="session_timeout",
            help="How long the secure session remains active before requiring re-authentication.",
        )

        if secure_endpoint_url:
            st.info(
                "🔒 Secure client will establish an encrypted communication channel with end-to-end encryption "
                "and optional attestation verification for enhanced security."
            )
        else:
            st.warning(
                "⚠️ Please configure the secure endpoint URL to use the secure client."
            )

    else:
        # Clear these session state variables when a different provider is selected
        if "use_web_search" in st.session_state:
            st.session_state.use_web_search = False
        if "include_search_queries" in st.session_state:
            st.session_state.include_search_queries = False
        if "max_web_search_uses" in st.session_state:
            st.session_state.max_web_search_uses = 5
        if "use_caching" in st.session_state:
            st.session_state.use_caching = False
        if "use_code_interpreter" in st.session_state:
            st.session_state.use_code_interpreter = False
        if "thinking_budget" in st.session_state:
            st.session_state.thinking_budget = 0

    # Local model provider selection
    st.subheader("Local Model Provider")
    local_provider_options = ["Ollama"]
    if mlx_available:
        local_provider_options.append("MLX")
    if cartesia_available:
        local_provider_options.append("Cartesia-MLX")
    if transformers_available:
        local_provider_options.append("Transformers")
    
    # Check if ModularClient is available
    modular_available = False
    try:
        from minions.clients.modular import ModularClient
        modular_available = True
        local_provider_options.append("Modular")
    except ImportError:
        pass

    local_provider = st.radio(
        "Select Local Provider",
        options=local_provider_options,
        horizontal=True,
        index=0,
    )

    # Add note about Cartesia-MLX installation if selected
    if local_provider == "Cartesia-MLX":
        st.info(
            "⚠️ Cartesia-MLX requires additional installation. Please check the README (see Setup Section) for instructions on how to install the cartesia-mlx package."
        )

    if local_provider == "MLX":
        st.info(
            "⚠️ MLX requires additional installation. Please check the README (see Setup Section) for instructions on how to install the mlx-lm package."
        )

    if local_provider == "Transformers":
        st.info(
            "⚠️ Transformers requires additional installation. Please install it with `pip install transformers torch`."
        )

        # Check if HF_TOKEN is set in environment or provide input field
        hf_token_env = os.getenv("HF_TOKEN", "")
        hf_token = st.text_input(
            "HuggingFace Token (optional for public models)",
            type="password",
            help="Required for accessing gated models on HuggingFace",
        )

        # Validate the token if provided
        if hf_token:
            is_valid, msg = validate_huggingface_token(hf_token)
            if is_valid:
                st.success("**✓ Valid HuggingFace token format.**")
                # Store the token in environment for this session
                os.environ["HF_TOKEN"] = hf_token
            else:
                st.error(f"**✗ Invalid HuggingFace token.** {msg}")
        elif hf_token_env:
            st.success("**✓ Using HuggingFace token from environment variables.**")
        else:
            st.warning(
                "No HuggingFace token provided. Gated models may not be accessible."
            )

    if local_provider == "Modular":
        st.info(
            "⚠️ Modular MAX requires installation. Please install it following the instructions at https://docs.modular.com/max/packages"
        )

        # Weights path configuration
        modular_weights_path = st.text_input(
            "Weights Path",
            value=st.session_state.get("modular_weights_path", "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/smollm2-1.7b-instruct-q4_k_m.gguf"),
            key="modular_weights_path",
            help="Path to model weights (can be local path or HuggingFace repo path)",
            placeholder="HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/smollm2-1.7b-instruct-q4_k_m.gguf",
        )
        
        # Server configuration
        modular_verbose = st.toggle(
            "Verbose Logging",
            value=st.session_state.get("modular_verbose", False),
            key="modular_verbose",
            help="Enable verbose logging for the Modular MAX server"
        )
        
       
        # Check if MAX CLI is available
        try:
            import subprocess
            result = subprocess.run(['max', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                st.success("**✓ Modular MAX CLI is available.** You're good to go!")
            else:
                st.error("**✗ Modular MAX CLI not found.** Please install Modular MAX.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            st.error("**✗ Modular MAX CLI not found.** Please install Modular MAX following the instructions at https://docs.modular.com/max/packages")

    # Protocol selection
    st.subheader("Protocol")

    # Set a default protocol value
    protocol = "Minion"  # Default protocol

    if selected_provider in [
        "OpenAI",
        "AzureOpenAI",
        "Together",
        "OpenRouter",
        "Anthropic",
        "DeepSeek",
        "SambaNova",
        "LlamaAPI",
    ]:  # Added LlamaAPI and Anthropic to the list
        protocol_options = [
            "Minion",
            "Minions",
            "Minions-MCP",
            "Minion-CUA",
            "DeepResearch",
        ]
        protocol = st.segmented_control(
            "Communication protocol", options=protocol_options, default="Minion"
        )
        print(protocol)
    else:
        # For providers that don't support all protocols, show a message and use the default
        st.info(f"The {selected_provider} provider only supports the Minion protocol.")

    # Add privacy mode toggle when Minion protocol is selected
    if protocol == "Minion":
        # privacy_mode = st.toggle(
        #     "Privacy Mode",
        #     value=False,
        #     help="When enabled, worker responses will be filtered to remove potentially sensitive information",
        # )
        privacy_mode = False
        multi_turn_mode = st.toggle(
            "Multi-Turn Mode",
            value=False,
            help="When enabled, the assistant will remember previous interactions in the conversation",
        )

        if multi_turn_mode:
            max_history_turns = st.slider(
                "Max History Turns",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of conversation turns to remember",
            )
        else:
            max_history_turns = 0
    else:
        privacy_mode = False
        multi_turn_mode = False
        max_history_turns = 0

    if protocol == "Minions":
        use_bm25 = st.toggle(
            "Smart Retrieval",
            value=False,
            help="When enabled, only the most relevant chunks of context will be examined by minions, speeding up execution",
        )
        max_jobs_per_round = st.number_input(
            "Max Jobs/Round",
            min_value=1,
            max_value=2048,
            value=2048,
            step=1,
            help="Maximum number of jobs to run per round for Minions protocol",
        )

        # Add chunking functions dropdown
        chunking_functions = {
            "Chunk by Section (Default)": "chunk_by_section",
            "Chunk by Page": "chunk_by_page",
            "Chunk by Paragraph": "chunk_by_paragraph",
            "Chunk by Code": "chunk_by_code",
            "Chunk by Function and Class": "chunk_by_function_and_class",
        }

        selected_chunking_function = st.selectbox(
            "Chunking Method",
            options=list(chunking_functions.keys()),
            index=0,
            help="Select how to split the context into chunks for processing",
        )

        # Store the selected chunking function in session state
        chunk_fn = chunking_functions[selected_chunking_function]
        st.session_state.chunk_fn = chunk_fn

        # Add a help expander to explain chunking methods
        with st.expander("ℹ️ About Chunking Methods", expanded=False):
            st.markdown(
                """
            ### Chunking Methods
            
            **Chunk by Section:** Splits text by section headings (e.g., "## Section Title"). Best for well-structured documents with clear section markers.
            
            **Chunk by Page:** Divides text by page boundaries. Useful for PDFs or documents with clear page breaks.
            
            **Chunk by Paragraph:** Breaks text into paragraphs. Good for most general text documents.
            
            **Chunk by Code:** Splits code files by functions. Optimized for programming source code.
            
            **Chunk by Function and Class:** More advanced code splitting that preserves class context. Best for object-oriented code.
            
            Choose the method that best matches your document's structure for optimal processing.
            """
            )
    else:
        use_bm25 = False
        max_jobs_per_round = 2048
        st.session_state.chunk_fn = "chunk_by_section"  # Default value

    # Add MCP server selection when Minions-MCP is selected
    if protocol == "Minions-MCP":
        # Add disclaimer about mcp.json configuration
        st.warning(
            "**Important:** To use Minions-MCP, make sure your `mcp.json` file is properly configured with your desired MCP servers. "
        )

        # Initialize MCP config manager to get available servers
        mcp_config_manager = MCPConfigManager()
        available_servers = mcp_config_manager.list_servers()

        if available_servers:
            mcp_server_name = st.selectbox(
                "MCP Server",
                options=available_servers,
                index=0 if "filesystem" in available_servers else 0,
            )
            # Store the selected server name in session state
            st.session_state.mcp_server_name = mcp_server_name
        else:
            st.warning(
                "No MCP servers found in configuration. Please check your MCP configuration."
            )
            mcp_server_name = "filesystem"  # Default fallback
            st.session_state.mcp_server_name = mcp_server_name

    # Model Settings
    st.subheader("Model Settings")

    # Create two columns for local and remote model settings
    local_col, remote_col = st.columns(2)

    # Local model settings
    with local_col:
        st.markdown("### Local Model")
        st.image("assets/minion_resized.jpg", use_container_width=True)

        # Show different model options based on local provider selection
        if local_provider == "MLX":
            local_model_options = {
                "Llama-3.2-3B-Instruct-4bit (Recommended)": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "gemma-3-4b-it-qat-bf16": "mlx-community/gemma-3-4b-it-qat-bf16",
                "gemma-3-1b-it-qat-bf16": "mlx-community/gemma-3-1b-it-qat-bf16",
                "Qwen2.5-7B-8bit": "mlx-community/Qwen2.5-7B-8bit",
                "Qwen2.5-3B-8bit": "mlx-community/Qwen2.5-3B-8bit",
                "Llama-3.2-3B-Instruct-8bit": "mlx-community/Llama-3.2-3B-Instruct-8bit",
                "Llama-3.1-8B-Instruct": "mlx-community/Llama-3.1-8B-Instruct",
            }
        elif local_provider == "Cartesia-MLX":
            local_model_options = {
                "Llamba-8B-8bit (Recommended)": "cartesia-ai/Llamba-8B-8bit-mlx",
                "Llamba-1B-4bit": "cartesia-ai/Llamba-1B-4bit-mlx",
                "Llamba-3B-4bit": "cartesia-ai/Llamba-3B-4bit-mlx",
            }
        elif local_provider == "Transformers":
            local_model_options = {
                "MedGemma 4B (Recommended)": "google/medgemma-4b-it",
                "Mistral 7B Instruct v0.2 (Recommended)": "mistralai/Mistral-7B-Instruct-v0.2",
                "Llama 3 8B Instruct": "meta-llama/Llama-3.1-8B-Instruct",
                "Helium-1-2b": "kyutai/helium-1-2b",
                "Foundation-Sec-8B": "fdtn-ai/Foundation-Sec-8B",
            }
        elif local_provider == "Modular":
            local_model_options = {
                "HuggingFaceTB/SmolLM2-1.7B-Instruct (Recommended)": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "microsoft/Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
                "microsoft/Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
                "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            }
        else:  # Ollama            # Get available Ollama models
            available_ollama_models = OllamaClient.get_available_models()

            # Default recommended models list
            recommended_models = ["llama3.2", "llama3.1:8b", "qwen2.5:3b", "qwen2.5:7b"]

            # Initialize with default model options
            local_model_options = {
                "llama3.2 (Recommended)": "llama3.2",
                "llama3.1:8b (Recommended)": "llama3.1:8b",
                "deepseek-r1:latest": "deepseek-r1:latest",
                "gemma3:4b (multimodal)": "gemma3:4b",
                "qwen2.5vl:3b (multimodal)": "qwen2.5vl:3b",
                "phi4-mini-reasoning": "phi4-mini-reasoning",
                "Qwen/Qwen3-1.7B": "qwen3:1.7b",
                "Qwen/Qwen3-0.6B": "qwen3:0.6b",
                "Qwen/Qwen3-4B": "qwen3:4b",
                "sarvam-2b": "gaganyatri/sarvam-2b-v0.5:latest",
                "gemma3:4b-it-qat": "gemma3:4b-it-qat",
                "gemma3:1b-it-qat": "gemma3:1b-it-qat",
                "deepcoder:1.5b": "deepcoder:1.5b",
                "llama3.2:1b": "llama3.2:1b",
                "gemma3:4b": "gemma3:4b",
                "granite3.2-vision": "granite3.2-vision",
                "phi4": "phi4",
                "qwen2.5:1.5b": "qwen2.5:1.5b",
                "qwen2.5:3b (Recommended)": "qwen2.5:3b",
                "qwen2.5:7b (Recommended)": "qwen2.5:7b",
                "qwen2.5:14b": "qwen2.5:14b",
                "mistral7b": "mistral7b",
                "deepseek-r1:1.5b": "deepseek-r1:1.5b",
                "deepseek-r1:7b": "deepseek-r1:7b",
                "deepseek-r1:8b": "deepseek-r1:8b",
            }

            # Add any additional available models from Ollama that aren't in the default list
            if available_ollama_models:
                for model in available_ollama_models:
                    model_key = model
                    if model in recommended_models:
                        # If it's a recommended model but not in defaults, add with (Recommended)
                        if model not in local_model_options.values():
                            model_key = f"{model} (Recommended)"
                    # Add the model if it's not already in the options
                    if model not in local_model_options.values():
                        local_model_options[model_key] = model

        local_model_display = st.selectbox(
            "Model", options=list(local_model_options.keys()), index=0
        )
        local_model_name = local_model_options[local_model_display]
        st.session_state.current_local_model = local_model_name

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
            # Set default temperature to 0.001 for Cartesia models
            local_temperature = 0.001 if local_provider == "Cartesia-MLX" else 0.0
            local_max_tokens = 4096

    # Remote model settings
    with remote_col:
        st.markdown("### Remote Model")
        st.image("assets/gru_resized.jpg", use_container_width=True)

        # If MLX is selected, use the same models for remote
        if selected_provider == "OpenAI":
            model_mapping = {
                "gpt-4o (Recommended)": "gpt-4o",
                "gpt-4.1": "gpt-4.1",
                "gpt-4.1-mini": "gpt-4.1-mini",
                "gpt-4.1-nano": "gpt-4.1-nano",
                "gpt-4o-mini": "gpt-4o-mini",
                "o3": "o3",
                "o4-mini": "o4-mini",
                "o3-mini": "o3-mini",
                "o1": "o1",
                "o1-pro": "o1-pro",
            }
            default_model_index = 0
        elif selected_provider == "AzureOpenAI":
            model_mapping = {
                "gpt-4o (Recommended)": "gpt-4o",
                "gpt-4": "gpt-4",
                "gpt-4-turbo": "gpt-4-turbo",
                "gpt-35-turbo": "gpt-35-turbo",
            }
            default_model_index = 0
        elif selected_provider == "Gemini":
            model_mapping = {
                "gemini-2.0-pro-preview-06-05 (Recommended)": "gemini-2.5-pro-exp-06-05",
                "gemini-2.5-flash-preview": "gemini-2.5-flash-preview-05-20",
                "gemini-2.5-pro-preview": "gemini-2.5-pro-preview-05-06",
                "gemini-2.0-flash": "gemini-2.0-flash",
                "gemini-1.5-pro": "gemini-1.5-pro",
                "gemini-1.5-flash": "gemini-1.5-flash",
            }
            default_model_index = 0
        elif selected_provider == "SambaNova":
            model_mapping = {
                "Meta-Llama-3.1-8B-Instruct (Recommended)": "Meta-Llama-3.1-8B-Instruct",
                "DeepSeek-V3-0324": "DeepSeek-V3-0324",
                "Meta-Llama-3.3-70B-Instruct": "Meta-Llama-3.3-70B-Instruct",
                "Meta-Llama-3.1-405B-Instruct": "Meta-Llama-3.1-405B-Instruct",
                "Meta-Llama-3.1-70B-Instruct": "Meta-Llama-3.1-70B-Instruct",
                "Meta-Llama-3.2-3B-Instruct": "Meta-Llama-3.2-3B-Instruct",
                "Meta-Llama-3.2-1B-Instruct": "Meta-Llama-3.2-1B-Instruct",
                "Llama-3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
                "Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
                "Meta-Llama-Guard-3-8B": "Meta-Llama-Guard-3-8B",
                "Llama-3.1-Tulu-3-405B": "Llama-3.1-Tulu-3-405B",
                "Llama-3.1-Swallow-8B-Instruct-v0.3": "Llama-3.1-Swallow-8B-Instruct-v0.3",
                "Llama-3.1-Swallow-70B-Instruct-v0.3": "Llama-3.1-Swallow-70B-Instruct-v0.3",
                "DeepSeek-R1": "DeepSeek-R1",
                "DeepSeek-R1-Distill-Llama-70B": "DeepSeek-R1-Distill-Llama-70B",
                "E5-Mistral-7B-Instruct": "E5-Mistral-7B-Instruct",
                "Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
                "Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
                "QwQ-32B": "QwQ-32B",
                "Qwen2-Audio-7B-Instruct": "Qwen2-Audio-7B-Instruct",
            }
            default_model_index = 0
        elif selected_provider == "OpenRouter":
            model_mapping = {
                "Claude 3.5 Sonnet (Recommended)": "anthropic/claude-3.5-sonnet",
                "claude 3.7 Sonnet Latest": "anthropic/claude-3-7-sonnet-latest",
                "Claude 3 Opus": "anthropic/claude-3-opus",
                "GPT-4o": "openai/gpt-4o",
                "Mistral Large": "mistralai/mistral-large",
                "Llama 3 70B": "meta-llama/llama-3-70b-instruct",
                "Gemini 1.5 Pro": "google/gemini-1.5-pro",
            }
            default_model_index = 0
        elif selected_provider == "Anthropic":
            model_mapping = {
                "Claude 4 Opus (Recommended)": "claude-opus-4-20250514",
                "Claude 4 Sonnet (Recommended)": "claude-sonnet-4-20250514",
                "claude-3-7-sonnet-latest (Recommended for web search)": "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest": "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest": "claude-3-5-haiku-latest",
                "claude-3-opus-latest": "claude-3-opus-latest",
            }
            default_model_index = 0
        elif selected_provider == "Together":
            model_mapping = {
                "DeepSeek-V3 (Recommended)": "deepseek-ai/DeepSeek-V3",
                "Qwen 2.5 72B (Recommended)": "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "Meta Llama 3.1 405B (Recommended)": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
                "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "QWQ-32B": "Qwen/QwQ-32B-Preview",
            }
            default_model_index = 0
        elif selected_provider == "Perplexity":
            model_mapping = {
                "sonar-pro (Recommended)": "sonar-pro",
                "sonar": "sonar",
                "sonar-reasoning": "sonar-reasoning",
                "sonar-reasoning-pro": "sonar-reasoning-pro",
                "sonar-deep-research": "sonar-deep-research",
            }
            default_model_index = 0
        elif selected_provider == "Groq":
            model_mapping = {
                "llama-3.3-70b-versatile (Recommended)": "llama-3.3-70b-versatile",
                "llama-3.3-70b-specdec": "llama-3.3-70b-specdec",
                "deepseek-r1-distill-llama-70b-specdec": "deepseek-r1-distill-llama-70b-specdec",
                "qwen-2.5-32b": "qwen-2.5-32b",
            }
            default_model_index = 0
        elif selected_provider == "DeepSeek":
            model_mapping = {
                "deepseek-chat (Recommended)": "deepseek-chat",
                "deepseek-reasoner": "deepseek-reasoner",
            }
            default_model_index = 0
        elif selected_provider == "LlamaAPI":
            model_mapping = {
                "Llama-4-Maverick-17B-128E-Instruct-FP8 (Recommended)": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "Llama-4-Scout-17B-16E-Instruct-FP8": "Llama-4-Scout-17B-16E-Instruct-FP8",
                "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
                "Llama-3.3-8B-Instruct": "Llama-3.3-8B-Instruct",
            }
            default_model_index = 0
        elif selected_provider == "Sarvam":
            model_mapping = {
                "sarvam-m (Recommended)": "sarvam-m",
                "sarvam-1b": "sarvam-1b",
                "sarvam-3b": "sarvam-3b",
            }
            default_model_index = 0
        elif selected_provider == "Secure":
            # Get secure endpoint URL from session state or environment
            secure_endpoint_url = st.session_state.get(
                "secure_endpoint_url"
            ) or os.getenv("SECURE_ENDPOINT_URL")
            verify_attestation = st.session_state.get("verify_attestation", True)
            session_timeout = st.session_state.get("session_timeout", 3600)
            model_mapping = {
                "gemma3-27B (Recommended)": "gemma3-27B",
            }
            default_model_index = 0

            if not secure_endpoint_url:
                st.error(
                    "Secure endpoint URL is required. Please set it in the configuration."
                )
                secure_endpoint_url = "https://localhost:8000"  # Default fallback

            try:
                from minions.clients.secure import SecureClient

                st.session_state.remote_client = SecureClient(
                    endpoint_url=secure_endpoint_url,
                    verify_attestation=verify_attestation,
                    session_timeout=session_timeout,
                )
            except ImportError:
                st.error(
                    "SecureClient is not available. Please ensure the secure module is properly installed."
                )
                # Fallback to OpenAI client
                remote_model_name = "gpt-4o"
                st.session_state.remote_client = OpenAIClient(
                    model_name=remote_model_name,
                    api_key=api_key,
                )
        else:
            model_mapping = {}
            default_model_index = 0

        remote_model_display = st.selectbox(
            "Model",
            options=list(model_mapping.keys()),
            index=default_model_index,
            key="remote_model_display",  # Changed to a different key name
        )
        remote_model_name = model_mapping[remote_model_display]
        st.session_state.current_remote_model = remote_model_name
        st.session_state.remote_model = (
            remote_model_name  # Store the actual model name for reference
        )

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

            # Replace slider with select box for reasoning effort
            reasoning_effort = st.selectbox(
                "Reasoning Effort",
                options=["low", "medium", "high"],
                index=1,  # Default to "medium"
                help="Controls how much effort the model puts into reasoning",
                key="reasoning_effort",
            )
        else:
            remote_temperature = 0.0
            remote_max_tokens = 4096
            reasoning_effort = "medium"  # Default reasoning effort

    voice_generation_enabled = False

    # Only try to import and initialize the voice generator if user enables it
    if voice_generation_enabled and voice_generation_available is None:
        try:
            from minions.utils.voice_generator import VoiceGenerator

            st.session_state.voice_generator = VoiceGenerator()
            voice_generation_available = st.session_state.voice_generator.csm_available

            if voice_generation_available:
                st.success("🔊 Minion voice generation is enabled!")
                st.info(
                    "Minions will speak their responses (limited to 500 characters)"
                )
            else:
                st.error("Voice generation could not be initialized")
                st.info("Make sure CSM-MLX is properly installed")
                voice_generation_enabled = False
        except ImportError:
            st.error(
                "Voice generation requires CSM-MLX. Install with: `pip install -e '.[csm-mlx]'`"
            )
            voice_generation_available = False
            voice_generation_enabled = False
    elif voice_generation_enabled and voice_generation_available is False:
        st.error("Voice generation is not available")
        st.info(
            "Make sure CSM-MLX is properly installed with: `pip install -e '.[csm-mlx]'`"
        )
        voice_generation_enabled = False
    elif voice_generation_enabled and voice_generation_available:
        st.success("🔊 Minion voice generation is enabled!")
        st.info("Minions will speak their responses (limited to 500 characters)")

    st.session_state.voice_generation_enabled = voice_generation_enabled

    if protocol == "Minion-CUA":

        st.sidebar.markdown(
            """
        ### 🤖 Computer Automation Mode
        
        In this mode, the AI assistant will actually control your computer to perform actions like:
        - Opening applications
        - Typing text
        - Clicking elements
        - Using keyboard shortcuts
        - Opening URLs
        
        This requires accessibility permissions to be granted to your terminal or application running this code.
        """
        )

        test_col1, test_col2 = st.sidebar.columns(2)

        # Add CUA documentation and examples
        with st.sidebar.expander("ℹ️ CUA Automation Sample Tasks", expanded=False):
            st.markdown(
                """
            ### Computer User Automation
            
            This Minion can help automate GUI tasks on your Mac. Here are some examples:
            
            **Basic Tasks:**
            - "Open Calculator and perform a calculation"
            - "Open TextEdit and type a short note"
            - "Open Safari and go to google.com"
            
            **Multi-step Tasks:**
            - "Please open Safari, navigate to gmail.com, and help me log in"
            - "Open Calculator, solve 5432 × 789, then paste the result in TextEdit"
            
            **Supported Actions:**
            - Opening applications
            - Typing text
            - Clicking UI elements
            - Keyboard shortcuts
            - Opening URLs
            
            **Allowed Applications:**
            """
            )

            # Display allowed applications
            st.markdown("- " + "\n- ".join(KEYSTROKE_ALLOWED_APPS))

            st.markdown(
                "**Note:** For security, some actions are restricted. The Minion will break complex tasks into small steps for confirmation."
            )


# -------------------------
#   Main app layout
# -------------------------
# if protocol == "Minions":
#     st.title("Minions!")
# else:
#     st.title("Minion!")

if protocol == "DeepResearch":
    # Check both session state and environment variables for API keys
    firecrawl_api_key = st.session_state.get("firecrawl_api_key") or os.getenv(
        "FIRECRAWL_API_KEY"
    )
    serpapi_key = st.session_state.get("serpapi_api_key") or os.getenv(
        "SERPAPI_API_KEY"
    )

    # Store the keys in session state if they came from env vars
    if firecrawl_api_key and not st.session_state.get("firecrawl_api_key"):
        st.session_state.firecrawl_api_key = firecrawl_api_key
    if serpapi_key and not st.session_state.get("serpapi_api_key"):
        st.session_state.serpapi_api_key = serpapi_key

    # Initialize clients for DeepResearch
    if (
        "local_client" not in st.session_state
        or "remote_client" not in st.session_state
        or "method" not in st.session_state
        or "current_protocol" not in st.session_state
        or st.session_state.current_protocol != protocol
    ):
        st.write("Initializing clients for DeepResearch protocol...")

        local_client, remote_client, method = initialize_clients(
            local_model_name,
            remote_model_name,
            selected_provider,
            local_provider,
            protocol,
            local_temperature,
            local_max_tokens,
            remote_temperature,
            remote_max_tokens,
            provider_key,
            num_ctx=4096,
            reasoning_effort=reasoning_effort,
        )

        # Update session state
        st.session_state.local_client = local_client
        st.session_state.remote_client = remote_client
        st.session_state.method = method
        st.session_state.current_protocol = protocol
        st.session_state.current_local_provider = local_provider
        st.session_state.current_remote_provider = selected_provider

    # Render UI
    render_deep_research_ui(minions_instance=st.session_state.method)

else:
    st.subheader("Context")
    text_input = st.text_area("Optionally paste text here", value="", height=150)

    st.markdown("Or upload context from a webpage")
    # Check if FIRECRAWL_API_KEY is set in environment or provided by user
    firecrawl_api_key_env = os.getenv("FIRECRAWL_API_KEY", "")

    # Display URL input and API key fields side by side
    c1, c2 = st.columns(2)
    with c2:
        # make the text input not visible as it is a password input
        firecrawl_api_key = st.text_input(
            "FIRECRAWL_API_KEY", type="password", key="firecrawl_api_key"
        )

    # Set the API key in environment if provided by user
    if firecrawl_api_key and firecrawl_api_key != firecrawl_api_key_env:
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

    # Only show URL input if API key is available
    with c1:
        if firecrawl_api_key:
            url_input = st.text_input("Or paste a URL here", value="")

        else:
            st.info("Set FIRECRAWL_API_KEY to enable URL scraping")
            url_input = ""

    uploaded_files = st.file_uploader(
        "Or upload PDF / TXT / Source code / Images (Not more than a 100 pages total!)",
        type=[
            "txt",
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "c",
            "cpp",
            "h",
            "hpp",
            "hxx",
            "cc",
            "cxx",
            "java",
            "js",
            "jsx",
            "ts",
            "tsx",
            "py",
            "rb",
            "go",
            "swift",
            "vb",
            "cs",
            "php",
            "pl",
            "pm",
            "perl",
            "sh",
            "bash",
        ],
        accept_multiple_files=True,
    )

    file_content = ""
    images = []
    # if url_input is not empty, scrape the url
    if url_input:
        # check if the FIRECRAWL_API_KEY is set
        if not os.getenv("FIRECRAWL_API_KEY"):
            st.error("FIRECRAWL_API_KEY is not set")
            st.stop()
        file_content = scrape_url(url_input)["markdown"]

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
                    # check if docling is installed
                    try:
                        import docling_core
                        from minions.utils.doc_processing import process_pdf_to_markdown

                        current_content = (
                            process_pdf_to_markdown(uploaded_file.read()) or ""
                        )
                    except:
                        current_content = (
                            extract_text_from_pdf(uploaded_file.read()) or ""
                        )
                elif file_type in ["png", "jpg", "jpeg"]:
                    image_bytes = uploaded_file.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    images.append(image_base64)
                    if st.session_state.current_local_model == "granite3.2-vision":
                        current_content = "file is an image"
                    else:
                        current_content = extract_text_from_image(image_base64) or ""
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

    if text_input and file_content:
        context = f"{text_input}\n## file upload:\n{file_content}"
        if doc_metadata:
            doc_metadata = f"Input: Text input and {doc_metadata[6:]}"  # Remove "Input: " from start
    elif text_input:
        context = text_input
        doc_metadata = f"Input: Text input only. Length: {len(text_input)} characters."
    else:
        context = file_content

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

    with st.expander("View Combined Context"):
        st.text(context)

    # Add required context description
    context_description = st.text_input(
        "One-sentence description of the context (Required)", key="context_description"
    )

    # -------------------------
    #  Chat-like user input
    # -------------------------
    user_query = st.chat_input(
        "Enter your query or request here...", key="persistent_chat"
    )

    # A container at the top to display final answer
    final_answer_placeholder = st.empty()

    if user_query:
        # Clear any existing placeholders when starting a new query
        if "placeholder_messages" in st.session_state:
            # First try to properly clear all existing placeholders
            for role, placeholder in st.session_state.placeholder_messages.items():
                try:
                    placeholder.empty()
                except Exception as e:
                    print(f"Warning: Could not clear placeholder for {role}: {e}")
            # Then reset the dictionary
            st.session_state.placeholder_messages = {}
        
        # Initialize query tracking to help with callback management
        st.session_state.current_query_id = time.time()
        
        # Validate context description is provided
        if not context_description.strip():
            st.error(
                "Please provide a one-sentence description of the context before proceeding."
            )
            st.stop()

        with st.status(f"Running {protocol} protocol...", expanded=True) as status:
            try:
                # Initialize clients first (only once) or if protocol or providers have changed
                if (
                    "local_client" not in st.session_state
                    or "remote_client" not in st.session_state
                    or "method" not in st.session_state
                    or "current_protocol" not in st.session_state
                    or "current_local_provider" not in st.session_state
                    or "current_remote_provider" not in st.session_state
                    or "current_remote_model" not in st.session_state
                    or "current_local_model" not in st.session_state
                    or st.session_state.current_protocol != protocol
                    or st.session_state.current_local_provider != local_provider
                    or st.session_state.current_remote_provider != selected_provider
                    or st.session_state.current_remote_model != remote_model_name
                    or st.session_state.current_local_model != local_model_name
                    or st.session_state.get("multi_turn_mode") != multi_turn_mode
                    or (
                        multi_turn_mode
                        and st.session_state.get("max_history_turns")
                        != max_history_turns
                    )
                ):

                    st.write(f"Initializing clients for {protocol} protocol...")

                    # Get MCP server name if using Minions-MCP
                    mcp_server_name = None
                    if protocol == "Minions-MCP":
                        mcp_server_name = st.session_state.get(
                            "mcp_server_name", "filesystem"
                        )

                    if local_provider == "Cartesia-MLX":
                        if local_temperature < 0.01:
                            local_temperature = 0.00001

                    # Get reasoning_effort from the widget value directly
                    if "reasoning_effort" in st.session_state:
                        reasoning_effort = st.session_state.reasoning_effort
                    else:
                        reasoning_effort = "medium"  # Default if not set

                    (
                        st.session_state.local_client,
                        st.session_state.remote_client,
                        st.session_state.method,
                    ) = initialize_clients(
                        local_model_name,
                        remote_model_name,
                        selected_provider,
                        local_provider,
                        protocol,
                        local_temperature,
                        local_max_tokens,
                        remote_temperature,
                        remote_max_tokens,
                        provider_key,
                        num_ctx,
                        mcp_server_name=mcp_server_name,
                        reasoning_effort=reasoning_effort,
                        multi_turn_mode=multi_turn_mode,
                        max_history_turns=max_history_turns,
                    )
                    # Store the current protocol and local provider in session state
                    st.session_state.current_protocol = protocol
                    st.session_state.current_local_provider = local_provider
                    st.session_state.current_remote_provider = selected_provider
                    st.session_state.current_remote_model = remote_model_name
                    st.session_state.current_local_model = local_model_name

                # Then run the protocol with pre-initialized clients
                output, setup_time, execution_time = run_protocol(
                    user_query,
                    context,
                    doc_metadata,
                    status,
                    protocol,
                    local_provider,
                    images,
                )

                status.update(
                    label=f"{protocol} protocol execution complete!", state="complete"
                )

                # Display final answer at the bottom with enhanced styling
                st.markdown("---")  # Add a visual separator
                # render the oriiginal query
                st.markdown("## 🚀 Query")
                st.info(user_query)
                st.markdown("## 🎯 Final Answer")
                st.info(output["final_answer"])

                # Timing info
                st.header("Runtime")
                total_time = setup_time + execution_time
                # st.metric("Setup Time", f"{setup_time:.2f}s", f"{(setup_time/total_time*100):.1f}% of total")

                # Create columns for timing metrics
                timing_cols = st.columns(4)

                # Display execution time metrics
                timing_cols[0].metric("Total Execution", f"{execution_time:.2f}s")

                # Display remote and local time metrics if available
                if "remote_time" in output and "local_time" in output:
                    remote_time = output["remote_time"]
                    local_time = output["local_time"]
                    other_time = output["other_time"]

                    # Calculate percentages
                    remote_pct = (remote_time / execution_time) * 100
                    local_pct = (local_time / execution_time) * 100
                    other_pct = (other_time / execution_time) * 100

                    timing_cols[1].metric(
                        "Remote Model Time",
                        f"{remote_time:.2f}s",
                        f"{remote_pct:.1f}% of total",
                    )

                    timing_cols[2].metric(
                        "Local Model Time",
                        f"{local_time:.2f}s",
                        f"{local_pct:.1f}% of total",
                    )

                    timing_cols[3].metric(
                        "Overhead Time",
                        f"{other_time:.2f}s",
                        f"{other_pct:.1f}% of total",
                    )

                    # Add a bar chart for timing visualization
                    timing_df = pd.DataFrame(
                        {
                            "Component": ["Remote Model", "Local Model", "Overhead"],
                            "Time (seconds)": [remote_time, local_time, other_time],
                        }
                    )
                    st.bar_chart(timing_df, x="Component", y="Time (seconds)")
                else:
                    timing_cols[1].metric("Execution Time", f"{execution_time:.2f}s")

                # Token usage for both protocols
                if "local_usage" in output and "remote_usage" in output:
                    st.header("Token Usage")
                    local_total = getattr(
                        output["local_usage"], "prompt_tokens", 0
                    ) + getattr(output["local_usage"], "completion_tokens", 0)
                    remote_total = getattr(
                        output["remote_usage"], "prompt_tokens", 0
                    ) + getattr(output["remote_usage"], "completion_tokens", 0)
                    c1, c2 = st.columns(2)
                    c1.metric(
                        f"{local_model_name} (Local) Total Tokens",
                        f"{local_total:,}",
                        f"Prompt: {getattr(output['local_usage'], 'prompt_tokens', 0):,}, "
                        f"Completion: {getattr(output['local_usage'], 'completion_tokens', 0):,}",
                    )
                    c2.metric(
                        f"{remote_model_name} (Remote) Total Tokens",
                        f"{remote_total:,}",
                        f"Prompt: {getattr(output['remote_usage'], 'prompt_tokens', 0):,}, "
                        f"Completion: {getattr(output['remote_usage'], 'completion_tokens', 0):,}",
                    )
                    # Convert to long format DataFrame for explicit ordering
                    df = pd.DataFrame(
                        {
                            "Model": [
                                f"Local: {local_model_name}",
                                f"Local: {local_model_name}",
                                f"Remote: {remote_model_name}",
                                f"Remote: {remote_model_name}",
                            ],
                            "Token Type": [
                                "Prompt Tokens",
                                "Completion Tokens",
                                "Prompt Tokens",
                                "Completion Tokens",
                            ],
                            "Count": [
                                getattr(output["local_usage"], "prompt_tokens", 0),
                                getattr(output["local_usage"], "completion_tokens", 0),
                                getattr(output["remote_usage"], "prompt_tokens", 0),
                                getattr(output["remote_usage"], "completion_tokens", 0),
                            ],
                        }
                    )
                    st.bar_chart(df, x="Model", y="Count", color="Token Type")

                    # Display cost information for OpenAI models
                    if (
                        selected_provider
                        in ["OpenAI", "AzureOpenAI", "DeepSeek", "LlamaAPI"]
                        and remote_model_name in API_PRICES[selected_provider]
                    ):
                        st.header("Remote Model Cost")
                        pricing = API_PRICES[selected_provider][remote_model_name]
                        prompt_cost = (
                            getattr(output["remote_usage"], "prompt_tokens", 0)
                            / 1_000_000
                        ) * pricing["input"]
                        completion_cost = (
                            getattr(output["remote_usage"], "completion_tokens", 0)
                            / 1_000_000
                        ) * pricing["output"]
                        total_cost = prompt_cost + completion_cost

                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "Prompt Cost",
                            f"${prompt_cost:.4f}",
                            f"{getattr(output['remote_usage'], 'prompt_tokens', 0):,} tokens (at ${pricing['input']:.2f}/1M)",
                        )
                        col2.metric(
                            "Completion Cost",
                            f"${completion_cost:.4f}",
                            f"{getattr(output['remote_usage'], 'completion_tokens', 0):,} tokens (at ${pricing['output']:.2f}/1M)",
                        )
                        col3.metric(
                            "Total Cost",
                            f"${total_cost:.4f}",
                            f"{remote_total:,} total tokens",
                        )

                # Display meta information for minions protocol
                if "meta" in output:
                    st.header("Meta Information")
                    for round_idx, round_meta in enumerate(output["meta"]):
                        st.subheader(f"Round {round_idx + 1}")
                        if "local" in round_meta:
                            st.write(f"Local jobs: {len(round_meta['local']['jobs'])}")
                        if "remote" in round_meta:
                            st.write(
                                f"Remote messages: {len(round_meta['remote']['messages'])}"
                            )

                # After displaying the final answer, show conversation history if multi-turn mode is enabled
                if (
                    protocol == "Minion"
                    and multi_turn_mode
                    and hasattr(st.session_state.method, "conversation_history")
                ):
                    st.header("Conversation History")

                    if (
                        hasattr(st.session_state.method.conversation_history, "turns")
                        and st.session_state.method.conversation_history.turns
                    ):
                        # Add a button to clear history
                        if st.button("Clear Conversation History"):
                            st.session_state.method.conversation_history.clear()
                            st.success("Conversation history cleared!")
                            st.rerun()

                        # Display conversation turns
                        for i, turn in enumerate(
                            st.session_state.method.conversation_history.turns
                        ):
                            st.markdown(f"### Turn {i+1}")
                            st.markdown(f"**User:** {turn.query}")
                            st.markdown(f"**Assistant:** {turn.remote_output}")
                            st.markdown("---")

                        # Show summary if available
                        if (
                            hasattr(
                                st.session_state.method.conversation_history, "summary"
                            )
                            and st.session_state.method.conversation_history.summary
                        ):
                            with st.expander("Conversation Summary", expanded=False):
                                st.markdown(
                                    st.session_state.method.conversation_history.summary
                                )
                    else:
                        st.info("No conversation history yet.")

                if (
                    protocol == "Minion-CUA"
                    and "action_history" in output
                    and output["action_history"]
                ):
                    st.header("Automation Actions")

                    # Create a DataFrame for better display
                    actions_data = []
                    for action in output["action_history"]:
                        action_type = action.get("action", "unknown")
                        app_name = action.get("app_name", "N/A")

                        # Format parameters based on action type
                        params = ""
                        if action_type == "type_keystrokes":
                            params = f"Keys: '{action.get('keys', '')}'"
                        elif action_type == "click_element":
                            element = action.get("element_desc", "")
                            coords = action.get("coordinates", [])
                            params = element if element else f"Coords: {coords}"
                        elif action_type == "key_combo":
                            params = "+".join(action.get("combo", []))
                        elif action_type == "open_url":
                            params = action.get("url", "")

                        explanation = action.get("explanation", "")
                        result = action.get("result", "")

                        actions_data.append(
                            {
                                "Type": action_type.replace("_", " ").title(),
                                "Application": app_name,
                                "Parameters": params,
                                "Explanation": explanation,
                                "Result": result,
                            }
                        )

                    if actions_data:
                        actions_df = pd.DataFrame(actions_data)
                        st.dataframe(actions_df, use_container_width=True)

                        # Create a visual timeline of actions
                        st.subheader("Action Timeline")
                        for i, action in enumerate(actions_data):
                            step = f"**Step {i+1}: {action['Type']} - {action['Application']}**"
                            st.markdown(step)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Parameters:**")
                                st.write(action["Parameters"])
                            with col2:
                                st.markdown("**Purpose:**")
                                st.write(action["Explanation"])
                            with col3:
                                st.markdown("**Result:**")
                                st.write(action["Result"])
                            st.markdown("---")  # Add a separator line

                        # Add a button to try an advanced workflow
                        if st.button("Try Gmail Login Workflow"):
                            with st.status(
                                "Starting Gmail login workflow...", expanded=True
                            ) as status:
                                workflow_task = """
                                Please help me log into Gmail. Here's what you need to do step by step:
                                1. Open Safari browser
                                2. Navigate to gmail.com
                                3. Wait for me to confirm when the login page is loaded
                                4. I'll provide my username and password separately for security
                                """

                                st.session_state.method(
                                    task=workflow_task,
                                    context=[
                                        "User needs to check their Gmail account."
                                    ],
                                    max_rounds=10,
                                    is_privacy=True,
                                )
                                status.update(
                                    label="Gmail workflow completed!", state="complete"
                                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
