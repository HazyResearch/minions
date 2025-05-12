# remote/worker_server.py
from flask import Flask, request, jsonify, Response, stream_with_context
from secure.utils.crypto_utils import *
from secure.remote.remote_model import run_model, initialize_model
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
    logger.info(
        f"🧠 Using SGLang with streaming enabled at endpoint: {args.sglang_endpoint}"
    )
else:
    os.environ["SGLANG_STREAMING"] = "false"
    logger.info(
        f"🧠 Using SGLang (non-streaming) with endpoint: {args.sglang_endpoint}"
    )

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

nonce = os.urandom(32)  # fresh each call
report, report_json, gpu_eat = create_attestation_report(
    "remote-worker", public_key, nonce
)
signature = sign_attestation(report_json, private_key)
logger.info("✅ SECURITY: Attestation report created and signed")

# Track sessions (by public key)
shared_keys = {}
nonces = {}
logger.info(
    "🔢 SECURITY: Initialized session tracking for key management and replay protection"
)


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
            "report": report,
            "report_json": report_json.decode(),
            "signature": signature,
            "public_key": serialize_public_key(public_key),
            "gpu_eat": gpu_eat,
            "nonce_b64": base64.b64encode(nonce).decode(),
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

    # Check if any messages contain image data
    has_image = False
    for msg in worker_messages:
        if msg.get("role") == "user" and "image_url" in msg:
            has_image = True
            image_url = msg["image_url"]
            # Log the first 50 characters of the image URL for debugging
            logger.info(f"🖼️ Message contains image data: {image_url[:50]}...")

    if has_image:
        logger.info("🧠 Processing message with image using SGLang")
    else:
        logger.info("🧠 Processing message without image using SGLang")
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
    
    # Check if any messages contain image data
    for msg in worker_messages:
        if msg.get("role") == "user" and "image_url" in msg:
            logger.info("🖼️ Message contains image data for streaming")

    # Check if any messages contain image data
    for msg in worker_messages:
        if msg.get("role") == "user" and "image_url" in msg:
            logger.info("🖼️ Message contains image data for streaming")

    def generate():
        # Create a counter for the nonce that's local to this function
        nonce_counter = initial_nonce

        # Use SGLang for streaming if enabled
        from secure.remote.remote_model import SGLangClient

        logger.info("🧠 Streaming response using SGLang")

        # Get the SGLang client
        client = SGLangClient.get_instance()

        # Get the streaming state
        state = client.stream_chat(worker_messages)
        # log the state
        logger.info(f"🧠 Streaming state: {state}")
        # Stream the response
        full_response = ""
        for chunk in state:
            new_text = chunk.choices[0].delta.content
            if new_text:
                full_response += new_text
                encrypted_chunk = encrypt_and_sign(
                    new_text, key, private_key, nonce_counter
                )
                nonce_counter += 1
                yield json.dumps(encrypted_chunk) + "\n"

        # Send an end-of-stream marker
        yield json.dumps({"eos": True}) + "\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    logger.info(f"🚀 Starting secure worker server on {args.host}:{args.port}")
    # Set debug=False for production environments
    app.run(host=args.host, port=args.port, debug=False)

