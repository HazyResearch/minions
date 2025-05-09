import json
import uuid
import requests
import logging
import time
from typing import List, Dict, Any, Optional, Generator, Callable

from secure.utils.crypto_utils import (
    generate_key_pair,
    create_attestation_report,
    sign_attestation,
    verify_attestation,
    derive_shared_key,
    encrypt_and_sign,
    decrypt_and_verify,
    serialize_public_key,
    deserialize_public_key,
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
        supervisor_att = requests.get(f"{self.supervisor_url}/attestation").json()
        self.supervisor_pub = deserialize_public_key(supervisor_att["public_key"])

        self.logger.info(
            "🔐 SECURITY: Verifying attestation report to ensure server identity and integrity"
        )

        if not verify_attestation(
            supervisor_att["report"],
            supervisor_att["report_json"].encode(),
            supervisor_att["signature"],
            self.supervisor_pub,
        ):
            self.logger.error(
                "🚨 SECURITY BREACH: Supervisor attestation verification failed"
            )
            raise RuntimeError("Supervisor attestation failed")

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

    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the supervisor and get a response"""
        if not self.is_initialized:
            self.initialize_secure_session()

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})

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
        self, message: str, callback: Callable[[str], None] = None
    ) -> Dict[str, Any]:
        """Send a message to the supervisor and get a streaming response"""
        if not self.is_initialized:
            self.initialize_secure_session()

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})

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

            # Ask if user wants to stream
            use_streaming = input("Use streaming? (y/n): ").lower() == "y"

            if use_streaming:
                print("Streaming response...")

                def print_chunk(chunk):
                    print(chunk, end="", flush=True)

                result = chat.send_message_stream(user_input, callback=print_chunk)
                print("\n")  # Add a newline after streaming completes
                print(
                    f"Streaming time: {result['time_spent']['total_streaming_time']:.3f}s"
                )
            else:
                print("Sending message securely...")
                result = chat.send_message(user_input)
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
