import json
import uuid
import requests
import logging
import time
import base64
import os
from typing import List, Dict, Any, Optional, Tuple

from urllib.parse import urlparse

from minions.usage import Usage
from minions.clients.base import MinionsClient


# Import crypto utilities from secure module
try:
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

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning(
        "Secure crypto utilities not available. SecureClient will not function properly."
    )


class SecureClient(MinionsClient):
    def __init__(
        self,
        endpoint_url: str,
        trusted_attestor_pem: str,
        model_name: str = "secure-model",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 30,
        verify_attestation: bool = True,
        session_timeout: int = 3600,  # 1 hour default session timeout
        **kwargs,
    ):
        """
        Initialize the Secure Client for encrypted communication with secure endpoints.

        Args:
            endpoint_url: URL of the secure endpoint
            model_name: Name of the model (for compatibility with other clients)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            timeout: Request timeout in seconds (default: 30)
            verify_attestation: Whether to verify endpoint attestation (default: True)
            session_timeout: Session timeout in seconds (default: 3600)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Client-specific configuration
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "Secure crypto utilities are required for SecureClient. "
                "Please ensure the secure module is properly installed."
            )

        self.endpoint_url = self._validate_endpoint_url(endpoint_url)
        self.supervisor_host = urlparse(endpoint_url).hostname
        self.supervisor_port = urlparse(endpoint_url).port

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.verify_attestation = verify_attestation
        self.session_timeout = session_timeout

        # Set up logging
        self.logger = logging.getLogger("SecureClient")
        self.logger.setLevel(logging.INFO)

        # Secure communication state
        self.session_id = None
        self.shared_key = None
        self.endpoint_pub = None
        self.local_priv = None
        self.local_pub = None
        self.nonce = 1000
        self.is_initialized = False
        self.session_start_time = None

        if not trusted_attestor_pem:
            raise ValueError(
                "You must provide a path to the trusted attesator public key. Please provide an attestator certificate. If you would like to use the hosted endpoint, request the PEM file by filling out this form: https://forms.gle/21ZAH9NqkehUwbiQ7"
            )

        self.trusted_attestor_hash = get_pem_hash(trusted_attestor_pem)

        print("🔒 SecureClient initialized for encrypted communication")

    def _validate_endpoint_url(self, supervisor_url: str) -> str:
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

        print(f"✅ SECURITY: Validated HTTPS supervisor URL: {supervisor_url}")
        return supervisor_url

    def estimate_tokens(
        self, messages: List[Dict[str, Any]], response_text: str = ""
    ) -> Tuple[int, int]:
        """
        Estimate the number of prompt tokens and completion tokens using character-based approximation.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_text: The response text to estimate completion tokens for

        Returns:
            Tuple of (prompt_tokens, completion_tokens)
        """
        # Character-based estimation (rough approximation: 4 chars ≈ 1 token)

        # Estimate prompt tokens from messages
        total_chars = 0
        for message in messages:
            for value in message.values():
                total_chars += len(str(value))

        # Add some overhead for message formatting (3 tokens per message)
        prompt_tokens = max(1, int(total_chars / 4) + len(messages) * 3)

        # Estimate completion tokens
        completion_tokens = max(0, int(len(response_text) / 4)) if response_text else 0

        return prompt_tokens, completion_tokens

    def _initialize_secure_session(self):
        """Set up the secure communication channel with attestation and key exchange"""

        if self.is_initialized and self._is_session_valid():
            return

        self.session_id = uuid.uuid4().hex
        self.session_start_time = time.time()
        print(f"🔒 Starting secure communication session {self.session_id}")

        # Fetch attestation from endpoint if verification is enabled

        if self.verify_attestation:
            print("🔐 SECURITY: Requesting attestation report from endpoint")

            try:
                attestation_response = requests.get(
                    f"{self.endpoint_url}/attestation", timeout=self.timeout
                )
                attestation_response.raise_for_status()
                endpoint_att = attestation_response.json()

                self.endpoint_pub = deserialize_public_key(
                    endpoint_att["public_key_worker"]
                )
                endpoint_nonce = base64.b64decode(endpoint_att["nonce_b64"])
                self.attesation_pub = deserialize_public_key(
                    endpoint_att["public_key_attestor"]
                )

                # Verify endpoint attestation
                verify_attestation_full(
                    report_json=endpoint_att["report_json"].encode(),
                    signature_b64=endpoint_att["signature"],
                    gpu_eat_json=endpoint_att["gpu_eat"],
                    public_key=self.attesation_pub,
                    expected_nonce=endpoint_nonce,
                    host=self.supervisor_host,
                    port=self.supervisor_port,
                    trusted_attestation_hash=self.trusted_attestor_hash,
                )
                print("✅ SECURITY: Endpoint attestation verification successful")

            except ValueError as e:
                self.logger.error("🚨 Endpoint attestation failed: %s", e)
                raise RuntimeError("Endpoint attestation failed") from e
            except requests.RequestException as e:
                self.logger.error("🚨 Failed to fetch attestation: %s", e)
                raise RuntimeError("Failed to fetch endpoint attestation") from e

        # Generate ephemeral keypair for the session
        print("🔑 SECURITY: Generating ephemeral key pair for perfect forward secrecy")
        self.local_priv, self.local_pub = generate_key_pair()
        self.shared_key = derive_shared_key(self.local_priv, self.endpoint_pub)

        print(
            "✅ SECURITY: Established shared secret key using Diffie-Hellman key exchange"
        )
        print("🔢 SECURITY: Initializing nonce counter for replay protection")

        self.is_initialized = True

    def _is_session_valid(self) -> bool:
        """Check if the current session is still valid"""
        if not self.session_start_time:
            return False
        return (time.time() - self.session_start_time) < self.session_timeout

    def _send_secure_message(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Send an encrypted message to the secure endpoint"""
        # Ensure secure session is initialized
        self._initialize_secure_session()

        # Encrypt and sign the message
        print(
            "🔒 SECURITY: Encrypting message with shared key and signing with private key"
        )
        message_json = json.dumps(messages)
        encrypted_payload = encrypt_and_sign(
            message_json, self.shared_key, self.local_priv, self.nonce
        )
        self.nonce += 1

        # Prepare request data
        request_data = {
            "peer_public_key": serialize_public_key(self.local_pub),
            "payload": encrypted_payload,
            "session_id": self.session_id,
        }

        print("📤 SECURITY: Sending encrypted and signed payload to endpoint")

        try:
            response = requests.post(
                f"{self.endpoint_url}/message",
                json=request_data,
                timeout=self.timeout,
            )
            # response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Request to secure endpoint failed: {str(e)}")
            raise RuntimeError(f"Failed to connect to secure endpoint: {str(e)}")

        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError(
                f"Endpoint returned invalid JSON response: {response.text}"
            )

        # Decrypt and verify the response
        print("📥 SECURITY: Decrypting and verifying endpoint response")
        decrypted_response = decrypt_and_verify(
            response_json, self.shared_key, self.endpoint_pub
        )

        print("✅ SECURITY: Message authentication and decryption successful")

        return {
            "response": decrypted_response,
        }

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the secure endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the secure endpoint

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            # Send the secure message
            response_data = self._send_secure_message(messages, **kwargs)

            # Extract the response content
            if "choices" in response_data:
                # OpenAI-style response format
                responses = [
                    choice["message"]["content"] for choice in response_data["choices"]
                ]
            elif "content" in response_data:
                # Simple response format
                responses = [response_data["content"]]
            elif "response" in response_data:
                # Alternative response format
                responses = [response_data["response"]]
            else:
                # Fallback - try to extract any text content
                responses = [str(response_data)]

            # Extract usage information from endpoint response
            usage_data = response_data.get("usage", {})

            # If endpoint provides usage information, use it
            if usage_data and (
                usage_data.get("prompt_tokens", 0) > 0
                or usage_data.get("completion_tokens", 0) > 0
            ):
                usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                )
                self.logger.debug("Using token usage from endpoint response")
            else:
                # Estimate tokens using our helper function
                response_text = responses[0] if responses else ""
                estimated_prompt_tokens, estimated_completion_tokens = (
                    self.estimate_tokens(messages, response_text)
                )

                usage = Usage(
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=estimated_completion_tokens,
                )
                print(
                    f"Estimated token usage - Prompt: {estimated_prompt_tokens}, Completion: {estimated_completion_tokens}"
                )

            return responses, usage

        except Exception as e:
            self.logger.error(f"Error during secure chat completion: {e}")
            raise

    def end_session(self):
        """End the secure session and clear all sensitive data"""
        if self.session_id:
            print(f"🔒 Ending secure session {self.session_id}")

            # Optionally notify the endpoint that the session is ending
            try:
                if self.is_initialized:
                    end_session_data = {"action": "end_session"}
                    self._send_secure_message([], **end_session_data)
            except Exception as e:
                self.logger.warning(f"Failed to notify endpoint of session end: {e}")

        # Clear all sensitive data
        self.shared_key = None
        self.endpoint_pub = None
        self.local_priv = None
        self.local_pub = None
        self.is_initialized = False
        self.session_start_time = None

        print("🔒 Secure session terminated and sensitive data cleared")
        self.session_id = None

    def __del__(self):
        """Cleanup when the client is destroyed"""
        if hasattr(self, "session_id") and self.session_id:
            self.end_session()
