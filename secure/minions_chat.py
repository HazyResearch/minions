import json
import uuid
import requests
import logging
import time
import base64
import os
import mimetypes
from typing import List, Dict, Any, Optional, Callable, Tuple
from urllib.parse import urlparse

from secure.utils.crypto_utils import (
    generate_key_pair,
    derive_shared_key,
    encrypt_and_sign,
    decrypt_and_verify,
    serialize_public_key,
    deserialize_public_key,
    verify_attestation_full,
    get_pem_hash,
)

from secure.utils.processing_utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    process_folder,
)

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class SecureMinionChat:
    def __init__(
        self,
        supervisor_url: str,
        trusted_attesation_pem: str, #file path
        vm_launch_measurement: str,
        system_prompt: str = None,
    ):
        self.logger = logger
        self.supervisor_url = self._validate_supervisor_url(supervisor_url)
        self.supervisor_host = urlparse(supervisor_url).hostname
        self.supervisor_port = urlparse(supervisor_url).port
        self.trusted_attestation_file = trusted_attesation_pem
        self.vm_launch_measurement = open(vm_launch_measurement).read()
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.conversation_history = []
        self.shared_key = None
        self.supervisor_pub = None
        self.local_priv = None
        self.local_pub = None
        self.nonce = 1000
        self.session_id = None
        self.is_initialized = False

        if not trusted_attesation_pem:
            raise ValueError(
                "You must provide a path to the trusted attesator public key. Please provide an attestator certificate. If you would like to use the hosted endpoint, fill out this form: https://forms.gle/21ZAH9NqkehUwbiQ7"
            )
        if not vm_launch_measurement:
            raise ValueError(
                "You must provide a path to a text file containing a hash of the VM launch measurement. Please provide a file path to the VM launch measurement."
            )

        self.trusted_attesation_hash = get_pem_hash(self.trusted_attestation_file)
        self.pem_file = open(self.trusted_attestation_file).read()
        self.attestation_pub = deserialize_public_key(self.pem_file)

        self.logger.info(
            "🔒 Secure Minion Chat initialized with end-to-end encryption and attestation verification"
        )

    def _validate_supervisor_url(self, supervisor_url: str) -> str:
        """Validate that the supervisor URL uses HTTPS protocol for security"""
        if not supervisor_url:
            raise ValueError("Supervisor URL cannot be empty")

        parsed_url = urlparse(supervisor_url)

        if parsed_url.scheme != "https":
            raise ValueError(
                f"Supervisor URL must use HTTPS protocol for secure communication. "
                f"Got: {parsed_url.scheme}://{parsed_url.netloc}"
            )

        if not parsed_url.netloc:
            raise ValueError("Invalid supervisor URL format")

        self.logger.info(
            f"✅ SECURITY: Validated HTTPS supervisor URL: {supervisor_url}"
        )
        return supervisor_url

    def initialize_secure_session(self):
        """Set up the secure communication channel with attestation and key exchange"""
        if self.is_initialized:
            return

        self.session_id = uuid.uuid4().hex
        self.logger.info(f"🔒 Starting secure communication session {self.session_id}")

        # Fetch attestation from supervisor server
        self.logger.info("🔐 SECURITY: Requesting attestation report from supervisor")

        start_time = time.time()
        att = requests.get(f"{self.supervisor_url}/attestation").json()

        self.supervisor_pub = deserialize_public_key(att["public_key_worker"])
        nonce = base64.b64decode(att["nonce_b64"])

        try:
            verify_attestation_full(
                report_json=att["report_json"].encode(),
                signature_b64=att["signature"],
                gpu_eat_json=att["gpu_eat"],
                public_key=self.attestation_pub,
                expected_nonce=nonce,
                server_host=self.supervisor_host,
                server_port=self.supervisor_port,
                trusted_attestation_hash=self.trusted_attesation_hash,
                vm_launch_measurement=self.vm_launch_measurement,
            )
        except ValueError as e:
            logger.error("🚨  supervisor attestation failed: %s", e)
            raise RuntimeError("Supervisor attestation failed") from e

        attestation_time = time.time() - start_time

        self.logger.info(f"🔒 SECURITY: Attestation report took: {attestation_time}")
        self.logger.info("✅ SECURITY: Attestation verification successful")

        # Generate ephemeral keypair for the session
        self.logger.info(
            "🔑 SECURITY: Generating ephemeral key pair for perfect forward secrecy"
        )

        start_time = time.time()
        self.local_priv, self.local_pub = generate_key_pair()
        self.shared_key = derive_shared_key(self.local_priv, self.supervisor_pub)
        key_exchange_time = time.time() - start_time

        self.logger.info(f"🔒 SECURITY: Key exchange took: {key_exchange_time}")

        self.logger.info(
            "✅ SECURITY: Established shared secret key using Diffie-Hellman key exchange"
        )

        self.logger.info(
            "🔢 SECURITY: Initializing nonce counter for replay protection"
        )

        # Initialize conversation with system prompt if provided
        if self.system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": self.system_prompt}
            )

        self.is_initialized = True

        return {
            "session_id": self.session_id,
            "attestation_time": attestation_time,
            "key_exchange_time": key_exchange_time,
        }

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from a PDF file"""
        return extract_text_from_pdf(pdf_path, custom_logger=self.logger)

    def _extract_text_from_txt(self, txt_path: str) -> Optional[str]:
        """Extract text content from a .txt file"""
        return extract_text_from_txt(txt_path, custom_logger=self.logger)

    def _process_folder(self, folder_path: str) -> Tuple[str, Optional[str]]:
        """Process a folder containing text, PDF, and image files.

        Returns:
            Tuple containing:
            - Concatenated text content from all text and PDF files
            - Path to a selected image (or None if no images found)
        """
        return process_folder(folder_path, custom_logger=self.logger)

    def send_message(
        self,
        message: str,
        image_path: str = None,
        pdf_path: str = None,
        folder_path: str = None,
    ) -> Dict[str, Any]:
        """Send a message to the supervisor and get a response"""
        if not self.is_initialized:
            self.initialize_secure_session()

        # Process folder if provided
        folder_text_content = ""
        if folder_path:
            folder_text_content, selected_image = self._process_folder(folder_path)
            # Override image_path with selected image from folder (if any)
            if selected_image:
                image_path = selected_image

        # Process PDF if provided
        pdf_content = None
        if pdf_path:
            pdf_content = self._extract_text_from_pdf(pdf_path)

        # Build context from all text sources
        context_parts = []
        if folder_text_content:
            context_parts.append(folder_text_content)
        if pdf_content:
            context_parts.append(pdf_content)

        # Format the message with all content as context
        if context_parts:
            context_text = "\n\n".join(context_parts)
            message = f"Context:\n\n{context_text}\n\nQuery:{message}"

        # Create the message object
        message_obj = {"role": "user", "content": message}

        # Process image if provided
        if image_path:
            # Process and add the image to the message
            image_data = self._process_image(image_path)
            if image_data:
                message_obj["image_url"] = image_data

        # Add user message to conversation history
        self.conversation_history.append(message_obj)

        # Timing dictionary
        time_spent = {
            "message_encryption": 0,
            "message_transmission": 0,
            "message_decryption": 0,
        }

        # Encrypt and send message to supervisor
        self.logger.info(
            "🔒 SECURITY: Encrypting message with shared key and signing with private key"
        )

        start_time = time.time()
        encrypted_messages = encrypt_and_sign(
            json.dumps(self.conversation_history),
            self.shared_key,
            self.local_priv,
            self.nonce,
        )
        time_spent["message_encryption"] = time.time() - start_time

        self.nonce += 1
        self.logger.info("🔢 SECURITY: Incrementing nonce for replay protection")

        try:
            request_data = {
                "peer_public_key": serialize_public_key(self.local_pub),
                "payload": encrypted_messages,
            }
            self.logger.info(
                "📤 SECURITY: Sending encrypted and signed payload to supervisor"
            )

            start_time = time.time()
            response = requests.post(
                f"{self.supervisor_url}/message",
                json=request_data,
                timeout=30,
            )
            time_spent["message_transmission"] = time.time() - start_time

            if response.status_code != 200:
                self.logger.error(
                    f"Supervisor request failed with status code {response.status_code}"
                )
                self.logger.error(f"Response content: {response.text}")
                raise RuntimeError(
                    f"Supervisor request failed with status code {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request to supervisor failed: {str(e)}")
            raise RuntimeError(f"Failed to connect to supervisor: {str(e)}")

        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError(
                f"Supervisor returned invalid JSON response: {response.text}"
            )

        self.logger.info("📥 SECURITY: Decrypting and verifying supervisor response")

        start_time = time.time()
        decrypted_response = decrypt_and_verify(
            response_json, self.shared_key, self.supervisor_pub
        )
        time_spent["message_decryption"] = time.time() - start_time

        self.logger.info(
            "✅ SECURITY: Message authentication and decryption successful"
        )

        # Add supervisor response to conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": decrypted_response}
        )

        return {
            "response": decrypted_response,
            "time_spent": time_spent,
            "session_id": self.session_id,
        }

    def send_message_stream(
        self,
        message: str,
        image_path: str = None,
        pdf_path: str = None,
        folder_path: str = None,
        callback: Callable[[str], None] = None,
    ) -> Dict[str, Any]:
        """Send a message to the supervisor and get a streaming response"""
        if not self.is_initialized:
            self.initialize_secure_session()

        # Process folder if provided
        folder_text_content = ""
        if folder_path:
            folder_text_content, selected_image = self._process_folder(folder_path)
            # Override image_path with selected image from folder (if any)
            if selected_image:
                image_path = selected_image

        # Process PDF if provided
        pdf_content = None
        if pdf_path:
            pdf_content = self._extract_text_from_pdf(pdf_path)

        # Build context from all text sources
        context_parts = []
        if folder_text_content:
            context_parts.append(folder_text_content)
        if pdf_content:
            context_parts.append(pdf_content)

        # Format the message with all content as context
        if context_parts:
            context_text = "\n\n".join(context_parts)
            message = f"Context:\n\n{context_text}\n\nQuery:{message}"

        # Create the message object
        message_obj = {"role": "user", "content": message}

        # Process image if provided
        if image_path:
            # Process and add the image to the message
            image_data = self._process_image(image_path)
            if image_data:
                message_obj["image_url"] = image_data

        # Add user message to conversation history
        self.conversation_history.append(message_obj)

        # Timing dictionary
        time_spent = {
            "message_encryption": 0,
            "message_transmission": 0,
            "total_streaming_time": 0,
        }

        # Encrypt and send message to supervisor
        self.logger.info(
            "🔒 SECURITY: Encrypting message with shared key and signing with private key"
        )

        start_time = time.time()
        encrypted_messages = encrypt_and_sign(
            json.dumps(self.conversation_history),
            self.shared_key,
            self.local_priv,
            self.nonce,
        )
        time_spent["message_encryption"] = time.time() - start_time

        self.nonce += 1
        self.logger.info("🔢 SECURITY: Incrementing nonce for replay protection")

        try:
            request_data = {
                "peer_public_key": serialize_public_key(self.local_pub),
                "payload": encrypted_messages,
            }
            self.logger.info(
                "📤 SECURITY: Sending encrypted and signed payload to supervisor for streaming"
            )

            start_time = time.time()

            # Use a streaming request
            with requests.post(
                f"{self.supervisor_url}/message_stream",
                json=request_data,
                timeout=60,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    self.logger.error(
                        f"Supervisor streaming request failed with status code {response.status_code}"
                    )
                    self.logger.error(f"Response content: {response.text}")
                    raise RuntimeError(
                        f"Supervisor streaming request failed with status code {response.status_code}: {response.text}"
                    )

                # Process the streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line)

                            # Check if this is the end-of-stream marker
                            if isinstance(chunk_data, dict) and chunk_data.get(
                                "eos", False
                            ):
                                break

                            # Decrypt the chunk
                            decrypted_chunk = decrypt_and_verify(
                                chunk_data, self.shared_key, self.supervisor_pub
                            )
                            # Append to the full response
                            full_response += decrypted_chunk

                            # Call the callback with the new chunk if provided
                            if callback:
                                callback(decrypted_chunk)
                        except Exception as e:
                            self.logger.error(
                                f"Error processing streaming chunk: {str(e)}"
                            )

            time_spent["total_streaming_time"] = time.time() - start_time
            self.logger.info("\n")
            self.logger.info("✅ SECURITY: Streaming completed successfully")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Streaming request to supervisor failed: {str(e)}")
            raise RuntimeError(
                f"Failed to connect to supervisor for streaming: {str(e)}"
            )

        # Add supervisor response to conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )

        return {
            "response": full_response,
            "time_spent": time_spent,
            "session_id": self.session_id,
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history"""
        return self.conversation_history

    def clear_conversation(self):
        """Clear the conversation history but keep the secure connection"""
        # Keep the system prompt if it exists
        if (
            self.conversation_history
            and self.conversation_history[0]["role"] == "system"
        ):
            system_prompt = self.conversation_history[0]
            self.conversation_history = [system_prompt]
        else:
            self.conversation_history = []

        self.logger.info("Conversation history cleared")

    def end_session(self):
        """End the secure session and clear all sensitive data"""
        self.conversation_history = []
        self.shared_key = None
        self.supervisor_pub = None
        self.local_priv = None
        self.local_pub = None
        self.is_initialized = False

        self.logger.info(f"🔒 Secure session {self.session_id} terminated")
        self.session_id = None

    def _process_image(self, image_path: str) -> Optional[str]:
        """Process an image file and return a data URL or None if processing fails"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None

            # Get the MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "application/octet-stream"  # Default MIME type

            # Read and encode the image
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

            self.logger.info(f"✅ Image processed successfully: {image_path}")
            return img_data

        except Exception as e:
            self.logger.error(f"❌ Error processing image: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Secure Minion Chat Client")
    parser.add_argument(
        "--supervisor_url", type=str, required=True, help="URL of the supervisor server"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="Optional system prompt to initialize the conversation",
    )
    parser.add_argument(
        "--trusted_attestation_pem",
        type=str,
        default="secure/attestation_public_key.pem",
        help="Path to the trusted attestation PEM file",
    )
    parser.add_argument(
        "--vm_launch_measurement_hashfile",
        type=str,
        default="secure/vm_launch_measurement.txt",
        help="Path to the VM launch measurement file",
    )
    args = parser.parse_args()

    chat = SecureMinionChat(
        args.supervisor_url, args.trusted_attestation_pem, args.vm_launch_measurement_hashfile, args.system_prompt
    )

    print(
        "🔒 Secure Minion Chat initialized. Type 'exit' to quit, 'clear' to clear conversation."
    )
    print("Establishing secure connection...")

    session_info = chat.initialize_secure_session()
    print(f"Secure session established with ID: {session_info['session_id']}")

    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == "exit":
                break

            if user_input.lower() == "clear":
                chat.clear_conversation()
                print("Conversation cleared.")
                continue

            # Ask if user wants to include an image, PDF, or folder
            attachment_type = None
            ask_for_attachment = (
                input("Include an attachment? (image/pdf/folder/none): ")
                .lower()
                .strip()
            )

            image_path = None
            pdf_path = None
            folder_path = None

            if ask_for_attachment == "image":
                image_path = input("Enter the path to the image file: ")
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    image_path = None
                else:
                    print(f"Including image: {image_path}")

            elif ask_for_attachment == "pdf":
                pdf_path = input("Enter the path to the PDF file: ")
                if not os.path.exists(pdf_path):
                    print(f"PDF file not found: {pdf_path}")
                    pdf_path = None
                else:
                    print(f"Including PDF context: {pdf_path}")

            elif ask_for_attachment == "folder":
                folder_path = input("Enter the path to the folder: ")
                if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                    print(f"Folder not found or not a directory: {folder_path}")
                    folder_path = None
                else:
                    print(f"Processing folder: {folder_path}")
                    # Count files by type
                    txt_count = len(glob.glob(os.path.join(folder_path, "*.txt")))
                    pdf_count = len(glob.glob(os.path.join(folder_path, "*.pdf")))
                    img_count = sum(
                        len(glob.glob(os.path.join(folder_path, f"*.{ext}")))
                        for ext in ["jpg", "jpeg", "png", "gif", "bmp"]
                    )
                    print(
                        f"Found {txt_count} text files, {pdf_count} PDF files, and {img_count} images."
                    )

            # Ask if user wants to stream
            use_streaming = True

            if use_streaming:
                print("Streaming response...")

                def print_chunk(chunk):
                    print(chunk, end="", flush=True)

                result = chat.send_message_stream(
                    user_input,
                    image_path=image_path,
                    pdf_path=pdf_path,
                    folder_path=folder_path,
                    callback=print_chunk,
                )
                print("\n")  # Add a newline after streaming completes
                print(
                    f"Streaming time: {result['time_spent']['total_streaming_time']:.3f}s"
                )
            else:
                print("Sending message securely...")
                result = chat.send_message(
                    user_input,
                    image_path=image_path,
                    pdf_path=pdf_path,
                    folder_path=folder_path,
                )
                print(f"\nAssistant: {result['response']}")
                print(
                    f"Message round-trip time: {sum(result['time_spent'].values()):.3f}s"
                )

            # Print detailed timing information
            print("\nDetailed timing:")
            for step, time_value in result["time_spent"].items():
                print(f"  - {step}: {time_value:.3f}s")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        chat.end_session()
        print("Secure session terminated.")
