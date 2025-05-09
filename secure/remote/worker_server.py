# remote/worker_server.py
from flask import Flask, request, jsonify, Response, stream_with_context
from minions.utils.crypto_utils import *
from minions.remote.remote_model import run_model, initialize_model
import os
import json
import logging
import argparse
import time

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Remote Worker Server")
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0 for all interfaces)",
)
parser.add_argument(
    "--port", type=int, default=5056, help="Port to listen on (default: 5006)"
)
parser.add_argument(
    "--key-path",
    type=str,
    default="worker_keys.json",
    help="Path to store worker keys (default: worker_keys.json)",
)
parser.add_argument(
    "--sglang-model",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Model to use with SGLang (default: Qwen/Qwen2.5-7B-Instruct)",
)
parser.add_argument(
    "--sglang-endpoint",
    type=str,
    default="http://localhost:5000",
    help="SGLang server endpoint (default: http://localhost:5000)",
)
parser.add_argument(
    "--streaming",
    action="store_true",
    help="Enable streaming mode for SGLang server",
)
args = parser.parse_args()


os.environ["USE_SGLANG"] = "true"
os.environ["SGLANG_ENDPOINT"] = args.sglang_endpoint
if args.streaming:
    os.environ["SGLANG_STREAMING"] = "true"
    logger.info(f"🧠 Using SGLang with streaming enabled at endpoint: {args.sglang_endpoint}")
else:
    os.environ["SGLANG_STREAMING"] = "false"
    logger.info(f"🧠 Using SGLang (non-streaming) with endpoint: {args.sglang_endpoint}")

app = Flask(__name__)
KEY_PATH = args.key_path

logger.info(
    f"🔒 Starting secure worker server with cryptographic protection on {args.host}:{args.port}"
)

# Persistent keypair for this remote node
if os.path.exists(KEY_PATH):
    logger.info("🔑 SECURITY: Loading existing cryptographic keys")
    with open(KEY_PATH, "r") as f:
        keys = json.load(f)
        private_key = serialization.load_pem_private_key(
            keys["private"].encode(), password=None
        )
        public_key = serialization.load_pem_public_key(keys["public"].encode())
else:
    logger.info("🔑 SECURITY: Generating new cryptographic key pair")
    private_key, public_key = generate_key_pair()
    with open(KEY_PATH, "w") as f:
        json.dump(
            {
                "private": private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ).decode(),
                "public": serialize_public_key(public_key),
            },
            f,
        )
    logger.info("✅ SECURITY: New keys generated and saved to disk")

# Attestation
logger.info("🔐 SECURITY: Creating attestation report for remote verification")
attestation_report, attestation_json = create_attestation_report(
    "remote-worker", public_key
)
attestation_signature = sign_attestation(attestation_json, private_key)
logger.info("✅ SECURITY: Attestation report created and signed")

# Track sessions (by public key)
shared_keys = {}
nonces = {}
logger.info(
    "🔢 SECURITY: Initialized session tracking for key management and replay protection"
)

# # Initialize the model if using vLLM or SGLang
# if os.environ.get("USE_VLLM", "false").lower() == "true":
#     logger.info("🧠 Initializing vLLM model at server startup")
#     initialize_model()
#     logger.info("✅ vLLM model initialized successfully")
# elif os.environ.get("USE_SGLANG", "false").lower() == "true":
logger.info("🧠 Initializing SGLang server and client at startup")
try:
    # Set the model path in environment
    os.environ["SGLANG_MODEL"] = args.sglang_model
    # Initialize the SGLang server and client
    initialize_model()
    logger.info("✅ SGLang initialization complete")
except Exception as e:
    logger.error(f"❌ Failed to initialize SGLang: {str(e)}")
    raise RuntimeError(f"SGLang initialization failed: {str(e)}")


@app.route("/attestation", methods=["GET"])
def attestation():
    logger.info(
        "📤 SECURITY: Sending attestation report to client for identity verification"
    )
    return jsonify(
        {
            "report": attestation_report,
            "report_json": attestation_json.decode(),
            "signature": attestation_signature,
            "public_key": serialize_public_key(public_key),
        }
    )

@app.route("/message", methods=["POST"])
def message():
    logger.info("📥 SECURITY: Received encrypted message")
    data = request.json
    peer_pub = deserialize_public_key(
        data["peer_public_key"]
    )  # Assuming this is the public key of the peer
    peer_id = serialize_public_key(peer_pub)

    if peer_id not in shared_keys:
        logger.info("🔑 SECURITY: New client connection - performing key exchange")
        shared_keys[peer_id] = derive_shared_key(private_key, peer_pub)
        nonces[peer_id] = 3000  # Arbitrary starting nonce for this peer
        logger.info(
            "✅ SECURITY: Established shared secret key using Diffie-Hellman key exchange"
        )
    else:
        logger.info("🔑 SECURITY: Using existing shared key for client")

    key = shared_keys[peer_id]
    nonce = nonces[peer_id]
    nonces[peer_id] += 1
    logger.info("🔢 SECURITY: Incrementing nonce for replay protection")

    payload = data["payload"]
    logger.info("🔓 SECURITY: Decrypting and verifying message authenticity")
    plaintext = decrypt_and_verify(payload, key, peer_pub)
    logger.info("✅ SECURITY: Message authentication and decryption successful")

    # Parse the JSON string to get the worker_messages list
    worker_messages = json.loads(plaintext)

    # worker_messages should already be in the correct format with 'role' and 'content' keys
    # Run local LLM with the full message history
    logger.info("🧠 Processing message with LLM")
    response_text = run_model(worker_messages)

    logger.info(
        "🔒 SECURITY: Encrypting response with shared key and signing with private key"
    )
    response_payload = encrypt_and_sign(response_text, key, private_key, nonce)
    logger.info("📤 SECURITY: Sending encrypted and signed response")
    return jsonify(response_payload)


@app.route("/message_stream", methods=["POST"])
def message_stream():
    logger.info("📥 SECURITY: Received encrypted streaming message request")
    data = request.json
    peer_pub = deserialize_public_key(
        data["peer_public_key"]
    )  # Assuming this is the public key of the peer
    peer_id = serialize_public_key(peer_pub)

    if peer_id not in shared_keys:
        logger.info("🔑 SECURITY: New client connection - performing key exchange")
        shared_keys[peer_id] = derive_shared_key(private_key, peer_pub)
        nonces[peer_id] = 3000  # Arbitrary starting nonce for this peer
        logger.info(
            "✅ SECURITY: Established shared secret key using Diffie-Hellman key exchange"
        )
    else:
        logger.info("🔑 SECURITY: Using existing shared key for client")

    key = shared_keys[peer_id]
    # Store the initial nonce value
    initial_nonce = nonces[peer_id]
    nonces[peer_id] += 1
    logger.info("🔢 SECURITY: Incrementing nonce for replay protection")

    payload = data["payload"]
    logger.info("🔓 SECURITY: Decrypting and verifying message authenticity")
    plaintext = decrypt_and_verify(payload, key, peer_pub)
    logger.info("✅ SECURITY: Message authentication and decryption successful")

    # Parse the JSON string to get the worker_messages list
    worker_messages = json.loads(plaintext)

    def generate():
        # Create a counter for the nonce that's local to this function
        nonce_counter = initial_nonce
        
        # Use SGLang for streaming if enabled
        if os.environ.get("USE_SGLANG", "false").lower() == "true":
            from minions.remote.remote_model import SGLangClient
            
            logger.info("🧠 Streaming response using SGLang")
            
            # Get the SGLang client
            client = SGLangClient.get_instance()
            
            # Get the streaming state
            state = client.stream_chat(worker_messages)
            
            # Stream the response
            full_response = ""
            for chunk in state.text_iter():
                if "<|im_start|>assistant" in chunk and "<|im_end|>" not in chunk:
                    # Skip the assistant tag
                    continue
                
                if "<|im_start|>" in chunk or "<|im_end|>" in chunk:
                    # Skip tags
                    continue
                
                # Extract new content
                if chunk and chunk not in full_response:
                    new_text = chunk
                    full_response += new_text
                    
                    # Encrypt each chunk with a new nonce
                    encrypted_chunk = encrypt_and_sign(new_text, key, private_key, nonce_counter)
                    nonce_counter += 1
                    
                    # Send the encrypted chunk
                    yield json.dumps(encrypted_chunk) + "\n"
            
            # Send an end-of-stream marker
            yield json.dumps({"eos": True}) + "\n"
            
        # Use vLLM for streaming if enabled
        elif os.environ.get("USE_VLLM", "false").lower() == "true":
            from minions.remote.remote_model import VLLMClient
            client = VLLMClient.get_instance()
            
            # Set up sampling parameters for streaming
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.0,
                top_p=1.0,
            )
            
            # Stream tokens from vLLM
            logger.info("🧠 Streaming response from vLLM")
            full_response = ""
            
            # Get the streaming outputs
            for output in client.model.chat(
                messages=worker_messages,
                sampling_params=sampling_params,
                stream=True
            ):
                if len(output.outputs) > 0:
                    # Get the new token(s)
                    new_text = output.outputs[0].text[len(full_response):]
                    full_response = output.outputs[0].text
                    
                    if new_text:
                        # Encrypt each chunk with a new nonce
                        encrypted_chunk = encrypt_and_sign(new_text, key, private_key, nonce_counter)
                        nonce_counter += 1
                        
                        # Send the encrypted chunk
                        yield json.dumps(encrypted_chunk) + "\n"
                        
            # Send an end-of-stream marker
            yield json.dumps({"eos": True}) + "\n"
        else:
            # For OpenAI, we don't have streaming yet, so just send the full response
            logger.info("🧠 Processing message with OpenAI (non-streaming)")
            from minions.remote.remote_model import run_model
            response_text = run_model(worker_messages)
            
            # Encrypt the full response
            encrypted_response = encrypt_and_sign(response_text, key, private_key, nonce_counter)
            yield json.dumps(encrypted_response) + "\n"
            yield json.dumps({"eos": True}) + "\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


if __name__ == "__main__":
    logger.info(f"🚀 Starting secure worker server on {args.host}:{args.port}")
    # Set debug=False for production environments
    app.run(host=args.host, port=args.port, debug=False)
