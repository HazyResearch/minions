import json
import uuid
import requests
import logging
import time
import base64
import os
from typing import List, Dict, Any, Optional, Generator, Callable
from urllib.parse import urlparse
import mimetypes

from secure.utils.crypto_utils import (
    generate_key_pair,
    derive_shared_key,
    encrypt_and_sign,
    decrypt_and_verify,
    serialize_public_key,
    deserialize_public_key,
    verify_attestation_full,
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
    def __init__(self, supervisor_url: str, system_prompt: str = None):
        self.logger = logger
        self.supervisor_url = supervisor_url
        self.supervisor_host = urlparse(supervisor_url).hostname
        self.supervisor_port = urlparse(supervisor_url).port
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.conversation_history = []
        self.shared_key = None
        self.supervisor_pub = None
        self.local_priv = None
        self.local_pub = None
        self.nonce = 1000
        self.session_id = None
        self.is_initialized = False

        self.logger.info(
            "🔒 Secure Minion Chat initialized with end-to-end encryption and attestation verification"
        )

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

        self.supervisor_pub = deserialize_public_key(att["public_key"])
        nonce = base64.b64decode(att["nonce_b64"])

        try:
            verify_attestation_full(
                report_json=att["report_json"].encode(),
                signature_b64=att["signature"],
                gpu_eat_json=att["gpu_eat"],
                public_key=self.supervisor_pub,
                expected_nonce=nonce,
                server_host=self.supervisor_host,
                server_port=self.supervisor_port,
            )
        except ValueError as e:
            logger.error("🚨  supervisor attestation failed: %s", e)
            raise RuntimeError("Supervisor attestation failed") from e

        attestation_time = time.time() - start_time
        self.logger.info("✅ SECURITY: Attestation verification successful")

        # Generate ephemeral keypair for the session
        self.logger.info(
            "🔑 SECURITY: Generating ephemeral key pair for perfect forward secrecy"
        )

        start_time = time.time()
        self.local_priv, self.local_pub = generate_key_pair()
        self.shared_key = derive_shared_key(self.local_priv, self.supervisor_pub)
        key_exchange_time = time.time() - start_time

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

    def send_message(self, message: str, image_path: str = None) -> Dict[str, Any]:
        """Send a message to the supervisor and get a response"""
        if not self.is_initialized:
            self.initialize_secure_session()

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
        callback: Callable[[str], None] = None,
    ) -> Dict[str, Any]:
        """Send a message to the supervisor and get a streaming response"""
        if not self.is_initialized:
            self.initialize_secure_session()

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
    args = parser.parse_args()

    chat = SecureMinionChat(args.supervisor_url, args.system_prompt)

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

            # Ask if user wants to include an image
            image_path = None
            include_image = input("Include an image? (y/n): ").lower() == "y"
            if include_image:
                image_path = input("Enter the path to the image file: ")
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    image_path = None
                else:
                    print(f"Including image: {image_path}")

            # Ask if user wants to stream
            use_streaming = input("Use streaming? (y/n): ").lower() == "y"

            if use_streaming:
                print("Streaming response...")

                def print_chunk(chunk):
                    print(chunk, end="", flush=True)

                result = chat.send_message_stream(
                    user_input, image_path, callback=print_chunk
                )
                print("\n")  # Add a newline after streaming completes
                print(
                    f"Streaming time: {result['time_spent']['total_streaming_time']:.3f}s"
                )
            else:
                print("Sending message securely...")
                result = chat.send_message(user_input, image_path)
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
