import json
import uuid
import requests
import logging
import time
import base64
import os
import mimetypes
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from urllib.parse import urlparse
import re

from secure.utils.crypto_utils import (
    generate_key_pair,
    derive_shared_key,
    encrypt_and_sign,
    decrypt_and_verify,
    serialize_public_key,
    deserialize_public_key,
    verify_attestation_full,
)

from secure.utils.processing_utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    process_folder,
)

from minions.usage import Usage
from minions.prompts.minion import (
    SUPERVISOR_INITIAL_PROMPT,
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
)

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def _escape_newlines_in_strings(json_str: str) -> str:
    """Escape newlines in JSON strings for better parsing."""
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    # Minimal fix: escape newlines only within quoted JSON strings.
    json_str = _escape_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        raise


class SecureMinionProtocol:
    def __init__(
        self,
        supervisor_url: str,
        local_client,
        max_rounds: int = 3,
        callback: Optional[Callable] = None,
        log_dir: str = "secure_minion_logs",
        system_prompt: str = None,
    ):
        """Initialize the Secure Minion Protocol with end-to-end encryption.

        Args:
            supervisor_url: URL of the supervisor server
            local_client: Client for the local worker model (e.g. OllamaClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
            log_dir: Directory for logging conversation history
            system_prompt: Optional system prompt for the worker
        """
        self.logger = logger
        self.supervisor_url = self._validate_supervisor_url(supervisor_url)
        self.supervisor_host = urlparse(supervisor_url).hostname
        self.supervisor_port = urlparse(supervisor_url).port
        self.local_client = local_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

        # Secure communication state (only for supervisor)
        self.session_id = None
        self.shared_key_supervisor = None
        self.supervisor_pub = None
        self.local_priv = None
        self.local_pub = None
        self.nonce = 1000
        self.is_initialized = False

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        self.logger.info(
            "🔒 Secure Minion Protocol initialized with end-to-end encryption and attestation verification"
        )

    def _validate_supervisor_url(self, supervisor_url: str) -> str:
        """Validate that the supervisor URL uses HTTPS protocol for security"""
        if not supervisor_url:
            raise ValueError("Supervisor URL cannot be empty")
        
        parsed_url = urlparse(supervisor_url)
        
        if parsed_url.scheme != 'https':
            raise ValueError(
                f"Supervisor URL must use HTTPS protocol for secure communication. "
                f"Got: {parsed_url.scheme}://{parsed_url.netloc}"
            )
        
        if not parsed_url.netloc:
            raise ValueError("Invalid supervisor URL format")
        
        self.logger.info(f"✅ SECURITY: Validated HTTPS supervisor URL: {supervisor_url}")
        return supervisor_url

    def initialize_secure_session(self):
        """Set up the secure communication channel with attestation and key exchange"""
        if self.is_initialized:
            return

        self.session_id = uuid.uuid4().hex
        self.logger.info(f"🔒 Starting secure communication session {self.session_id}")

        # Timing dictionary for setup
        setup_timing = {
            "attestation_exchange": 0,
            "key_exchange": 0,
        }

        # Fetch attestation from supervisor server only
        self.logger.info("🔐 SECURITY: Requesting attestation report from supervisor")

        start_time = time.time()
        
        # Get supervisor attestation
        supervisor_att = requests.get(f"{self.supervisor_url}/attestation").json()
        self.supervisor_pub = deserialize_public_key(supervisor_att["public_key"])
        supervisor_nonce = base64.b64decode(supervisor_att["nonce_b64"])

        # Verify supervisor attestation
        try:
            verify_attestation_full(
                report_json=supervisor_att["report_json"].encode(),
                signature_b64=supervisor_att["signature"],
                gpu_eat_json=supervisor_att["gpu_eat"],
                public_key=self.supervisor_pub,
                expected_nonce=supervisor_nonce,
                server_host=self.supervisor_host,
                server_port=self.supervisor_port,
            )
            self.logger.info("✅ SECURITY: Supervisor attestation verification successful")
        except ValueError as e:
            self.logger.error("🚨 Supervisor attestation failed: %s", e)
            raise RuntimeError("Supervisor attestation failed") from e

        setup_timing["attestation_exchange"] = time.time() - start_time

        # Generate ephemeral keypair for the session
        self.logger.info(
            "🔑 SECURITY: Generating ephemeral key pair for perfect forward secrecy"
        )

        start_time = time.time()
        self.local_priv, self.local_pub = generate_key_pair()
        self.shared_key_supervisor = derive_shared_key(self.local_priv, self.supervisor_pub)
        setup_timing["key_exchange"] = time.time() - start_time

        self.logger.info(
            "✅ SECURITY: Established shared secret key using Diffie-Hellman key exchange"
        )

        self.logger.info(
            "🔢 SECURITY: Initializing nonce counter for replay protection"
        )

        self.is_initialized = True
        return setup_timing

    def _process_image(self, image_path: str) -> Optional[str]:
        """Process an image file and return base64 encoded data or None if processing fails"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None

            # Get the MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            # Read and encode the image
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

            self.logger.info(f"✅ Image processed successfully: {image_path}")
            return img_data

        except Exception as e:
            self.logger.error(f"❌ Error processing image: {str(e)}")
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from a PDF file"""
        return extract_text_from_pdf(pdf_path, custom_logger=self.logger)

    def _extract_text_from_txt(self, txt_path: str) -> Optional[str]:
        """Extract text content from a .txt file"""
        return extract_text_from_txt(txt_path, custom_logger=self.logger)

    def _process_folder(self, folder_path: str):
        """Process a folder containing text, PDF, and image files."""
        return process_folder(folder_path, custom_logger=self.logger)

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds: Optional[int] = None,
        doc_metadata: Optional[Dict] = None,
        logging_id: Optional[str] = None,
        is_privacy: bool = False,
        images: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        pdf_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the secure minion protocol to answer a task.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files
            is_privacy: Whether privacy mode is enabled
            images: List of image paths (for compatibility)
            image_path: Single image path
            pdf_path: Path to PDF file to process
            folder_path: Path to folder to process

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """

        print("\n========== SECURE MINION TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Max rounds: {max_rounds or self.max_rounds}")
        print(f"Privacy enabled: {is_privacy}")
        print(f"Images provided: {True if (images or image_path) else False}")

        # Initialize timing metrics
        start_time = time.time()
        timing = {
            "setup": {},
            "rounds": [],
            "total_time": 0.0,
        }

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Initialize secure session
        if not self.is_initialized:
            timing["setup"] = self.initialize_secure_session()

        # Process additional inputs
        folder_text_content = ""
        if folder_path:
            folder_text_content, selected_image = self._process_folder(folder_path)
            if selected_image and not image_path:
                image_path = selected_image

        # Process PDF if provided
        pdf_content = None
        if pdf_path:
            pdf_content = self._extract_text_from_pdf(pdf_path)

        # Build context from all sources
        context_parts = list(context) if context else []
        if folder_text_content:
            context_parts.append(folder_text_content)
        if pdf_content:
            context_parts.append(pdf_content)

        # Join context sections
        context_str = "\n\n".join([f"### Document {i+1}\n{doc}" for i, doc in enumerate(context_parts)])
        print(f"Context length: {len(context_str)} characters")

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": context_str,
            "conversation": [],
            "generated_final_answer": "",
            "usage": {
                "remote": {},
                "local": {},
            },
            "timing": timing,
            "session_id": self.session_id,
        }

        # Initialize message histories and usage tracking
        supervisor_messages = []
        remote_usage = Usage()
        local_usage = Usage()

        # Format worker system prompt with context
        worker_system_prompt = WORKER_SYSTEM_PROMPT.format(
            context=context_str, task=task
        )

        # Initial question to Supervisor
        initial_prompt = SUPERVISOR_INITIAL_PROMPT.format(task=task)

        # Add initial prompt to supervisor messages
        supervisor_messages.append({"role": "user", "content": initial_prompt})

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "supervisor",
                "prompt": initial_prompt,
                "output": None,
            }
        )

        # Encrypt and send initial prompt to supervisor
        self.logger.info(
            "🔒 SECURITY: Encrypting message with shared key and signing with private key"
        )

        round_timing = {
            "supervisor_encryption": 0,
            "supervisor_transmission": 0,
            "supervisor_decryption": 0,
            "worker_call_time": 0,
        }

        start_time_round = time.time()
        initial_message = json.dumps([{"role": "user", "content": initial_prompt}])
        supervisor_payload = encrypt_and_sign(
            initial_message, self.shared_key_supervisor, self.local_priv, self.nonce
        )
        round_timing["supervisor_encryption"] = time.time() - start_time_round

        self.nonce += 1

        try:
            request_data = {
                "peer_public_key": serialize_public_key(self.local_pub),
                "payload": supervisor_payload,
            }
            self.logger.info(
                "📤 SECURITY: Sending encrypted and signed payload to supervisor"
            )

            start_time_round = time.time()
            sup_response = requests.post(
                f"{self.supervisor_url}/message",
                json=request_data,
                timeout=30,
            )
            round_timing["supervisor_transmission"] = time.time() - start_time_round

            if sup_response.status_code != 200:
                self.logger.error(
                    f"Supervisor request failed with status code {sup_response.status_code}"
                )
                raise RuntimeError(
                    f"Supervisor request failed with status code {sup_response.status_code}: {sup_response.text}"
                )
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request to supervisor failed: {str(e)}")
            raise RuntimeError(f"Failed to connect to supervisor: {str(e)}")

        try:
            sup_response_json = sup_response.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError(
                f"Supervisor returned invalid JSON response: {sup_response.text}"
            )

        self.logger.info("📥 SECURITY: Decrypting and verifying supervisor response")

        start_time_round = time.time()
        decrypted_sup = decrypt_and_verify(
            sup_response_json, self.shared_key_supervisor, self.supervisor_pub
        )
        round_timing["supervisor_decryption"] = time.time() - start_time_round

        self.logger.info(
            "✅ SECURITY: Message authentication and decryption successful"
        )

        # Add supervisor response to messages
        supervisor_messages.append({"role": "assistant", "content": decrypted_sup})

        # Update the last conversation entry with the output
        conversation_log["conversation"][-1]["output"] = decrypted_sup

        if self.callback:
            self.callback("supervisor", {"role": "assistant", "content": decrypted_sup})

        # Extract first question for worker
        try:
            supervisor_json = json.loads(decrypted_sup)
        except:
            supervisor_json = _extract_json(decrypted_sup)

        worker_query = supervisor_json["message"]

        # Initialize worker messages with system prompt
        worker_messages = [{"role": "system", "content": worker_system_prompt}]
        worker_messages.append({"role": "user", "content": worker_query})

        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "worker",
                "prompt": worker_query,
                "output": None,
            }
        )

        final_answer = None

        # Start interaction loop
        for round_idx in range(max_rounds):
            self.logger.info(f"🔄 Round {round_idx + 1}/{max_rounds}")

            # (1) Make a call to the local worker
            if self.callback:
                self.callback("worker", None, is_final=False)

            self.logger.info("🔧 LOCAL: Calling local worker model")

            start_time_round = time.time()
            worker_response, worker_usage, done_reason = self.local_client.chat(
                messages=worker_messages
            )
            round_timing["worker_call_time"] = time.time() - start_time_round

            self.logger.info(f"✅ LOCAL: Worker response received")

            local_usage += worker_usage

            # Add worker response to messages
            worker_messages.append({"role": "assistant", "content": worker_response[0]})

            # Update the last conversation entry with the output
            conversation_log["conversation"][-1]["output"] = worker_response[0]

            if self.callback:
                self.callback("worker", {"role": "assistant", "content": worker_response[0]})

            # (2) Make a call to the supervisor
            self.logger.info("📤 Sending worker response to supervisor")

            # Choose prompt template based on round
            if round_idx == max_rounds - 1:
                follow_up = SUPERVISOR_FINAL_PROMPT.format(response=worker_response[0])
            else:
                follow_up = SUPERVISOR_CONVERSATION_PROMPT.format(response=worker_response[0])

            # Add follow-up to supervisor messages
            supervisor_messages.append({"role": "user", "content": follow_up})

            # Add supervisor prompt to conversation log
            conversation_log["conversation"].append(
                {
                    "user": "supervisor",
                    "prompt": follow_up,
                    "output": None,
                }
            )

            self.logger.info(
                "🔒 SECURITY: Encrypting supervisor messages with shared key"
            )

            start_time_round = time.time()
            supervisor_messages_json = json.dumps(supervisor_messages)
            encrypted_supervisor_messages = encrypt_and_sign(
                supervisor_messages_json, self.shared_key_supervisor, self.local_priv, self.nonce
            )
            round_timing["supervisor_encryption"] += time.time() - start_time_round

            self.nonce += 1
            self.logger.info("🔢 SECURITY: Incrementing nonce for replay protection")

            try:
                request_data = {
                    "peer_public_key": serialize_public_key(self.local_pub),
                    "payload": encrypted_supervisor_messages,
                }

                start_time_round = time.time()
                sup_response = requests.post(
                    f"{self.supervisor_url}/message",
                    json=request_data,
                    timeout=30,
                )
                round_timing["supervisor_transmission"] += time.time() - start_time_round

                if sup_response.status_code != 200:
                    self.logger.error(
                        f"Supervisor request failed with status code {sup_response.status_code}"
                    )
                    raise RuntimeError(
                        f"Supervisor request failed with status code {sup_response.status_code}: {sup_response.text}"
                    )
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request to supervisor failed: {str(e)}")
                raise RuntimeError(f"Failed to connect to supervisor: {str(e)}")

            try:
                sup_response_json = sup_response.json()
            except requests.exceptions.JSONDecodeError:
                raise RuntimeError(
                    f"Supervisor returned invalid JSON response: {sup_response.text}"
                )

            self.logger.info(
                "📥 SECURITY: Decrypting and verifying supervisor response"
            )

            start_time_round = time.time()
            decrypted_sup = decrypt_and_verify(
                sup_response_json, self.shared_key_supervisor, self.supervisor_pub
            )
            round_timing["supervisor_decryption"] += time.time() - start_time_round

            self.logger.info(
                "✅ SECURITY: Message authentication and decryption successful"
            )

            # Add supervisor response to messages
            supervisor_messages.append({"role": "assistant", "content": decrypted_sup})

            # Update the last conversation entry with the output
            conversation_log["conversation"][-1]["output"] = decrypted_sup

            if self.callback:
                self.callback("supervisor", {"role": "assistant", "content": decrypted_sup})

            # Add round timing to the timing dictionary
            timing["rounds"].append(round_timing.copy())

            # (3) Check if the supervisor has requested additional info
            self.logger.info("🔍 Checking supervisor decision")
            try:
                supervisor_json = json.loads(decrypted_sup)
            except:
                supervisor_json = _extract_json(decrypted_sup)

            if supervisor_json["decision"] == "request_additional_info":
                self.logger.info(
                    "📋 Supervisor requested additional info. Continuing to next round."
                )
                worker_query = supervisor_json["message"]
                worker_messages.append({"role": "user", "content": worker_query})

                # Add worker prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "worker",
                        "prompt": worker_query,
                        "output": None,
                    }
                )

                # Reset round timing for next round
                round_timing = {
                    "supervisor_encryption": 0,
                    "supervisor_transmission": 0,
                    "supervisor_decryption": 0,
                    "worker_call_time": 0,
                }
                continue
            else:
                self.logger.info("✅ Found final answer")
                final_answer = supervisor_json["answer"]
                conversation_log["generated_final_answer"] = final_answer
                break

        if final_answer is None:
            self.logger.error("❌ Exhausted all rounds without finding a final answer.")
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Calculate total time
        end_time = time.time()
        timing["total_time"] = end_time - start_time

        # Add usage statistics to the log (placeholder for now)
        conversation_log["usage"]["remote"] = remote_usage
        conversation_log["usage"]["local"] = local_usage

        # Log the final result
        if logging_id:
            log_filename = f"{logging_id}_secure_minion.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}_secure.json"

        log_path = os.path.join(self.log_dir, log_filename)

        print(f"\n=== SAVING LOG TO {log_path} ===")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving log to {log_path}: {e}")

        self.logger.info(
            "🔒 SECURITY: Secure communication session completed successfully"
        )

        print("\n=== SECURE MINION TASK COMPLETED ===")

        return {
            "final_answer": final_answer,
            "text": final_answer,  # For compatibility
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "edge_usage": local_usage,  # For compatibility
            "log_file": log_path,
            "conversation_log": conversation_log,
            "timing": timing,
            "time_spent": timing,  # For compatibility
            "meta": {"log": {"conversation_id": self.session_id}},
            "session_id": self.session_id,
        }

    def end_session(self):
        """End the secure session and clear all sensitive data"""
        self.shared_key_supervisor = None
        self.supervisor_pub = None
        self.local_priv = None
        self.local_pub = None
        self.is_initialized = False

        self.logger.info(f"🔒 Secure session {self.session_id} terminated")
        self.session_id = None


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Secure Minion Protocol")
    parser.add_argument(
        "--supervisor_url", type=str, required=True, help="URL of the supervisor server"
    )
    parser.add_argument(
        "--local_client_type", type=str, default="ollama", help="Type of local client (ollama, openai, etc.)"
    )
    parser.add_argument(
        "--local_model", type=str, default="llama3.2", help="Local model name"
    )
    parser.add_argument(
        "--max_rounds", type=int, default=3, help="Maximum number of rounds"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="Optional system prompt for the worker",
    )
    args = parser.parse_args()

    # Initialize the local client based on type
    if args.local_client_type.lower() == "ollama":
        from minions.clients import OllamaClient
        local_client = OllamaClient(model=args.local_model)
    else:
        raise ValueError(f"Unsupported local client type: {args.local_client_type}")

    protocol = SecureMinionProtocol(
        supervisor_url=args.supervisor_url,
        local_client=local_client,
        max_rounds=args.max_rounds,
        system_prompt=args.system_prompt,
    )

    print("🔒 Secure Minion Protocol initialized.")
    print("Establishing secure connections...")

    try:
        while True:
            task = input("\nEnter your task (or 'exit' to quit): ")

            if task.lower() == "exit":
                break

            context_input = input("Enter context (or press Enter for none): ")
            context = [context_input] if context_input.strip() else []

            # Ask for attachments
            attachment_type = input("Include an attachment? (image/pdf/folder/none): ").lower().strip()

            image_path = None
            pdf_path = None
            folder_path = None

            if attachment_type == "image":
                image_path = input("Enter the path to the image file: ")
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    image_path = None

            elif attachment_type == "pdf":
                pdf_path = input("Enter the path to the PDF file: ")
                if not os.path.exists(pdf_path):
                    print(f"PDF file not found: {pdf_path}")
                    pdf_path = None

            elif attachment_type == "folder":
                folder_path = input("Enter the path to the folder: ")
                if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                    print(f"Folder not found or not a directory: {folder_path}")
                    folder_path = None

            print("Processing task securely...")
            result = protocol(
                task=task,
                context=context,
                image_path=image_path,
                pdf_path=pdf_path,
                folder_path=folder_path,
            )

            print(f"\n🎯 Final Answer: {result['final_answer']}")
            print(f"⏱️  Total time: {result['timing']['total_time']:.3f}s")
            print(f"📁 Log saved to: {result['log_file']}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        protocol.end_session()
        print("Secure session terminated.") 


### USAGE EXAMPLES
# python secure/minions_secure.py \
#     --supervisor_url https://supervisor.example.com \
#     --local_client_type ollama \
#     --local_model llama3.2


# from minions.clients import OllamaClient
# from secure.minions_secure import SecureMinionProtocol

# local_client = OllamaClient(model="llama3.2")
# protocol = SecureMinionProtocol(
#     supervisor_url="https://supervisor.example.com",
#     local_client=local_client
# )

# result = protocol(task="Analyze this document", context=["document content"])