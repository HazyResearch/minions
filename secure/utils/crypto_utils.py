import base64
import json
import os
import time
import hashlib
import hmac
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidSignature
import subprocess
import jwt  # requires pip install pyjwt

# ------------------ Key Management ------------------


def generate_key_pair():
    private_key = ec.generate_private_key(ec.SECP384R1())
    public_key = private_key.public_key()
    return private_key, public_key


def serialize_public_key(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def deserialize_public_key(pem_data):
    return serialization.load_pem_public_key(pem_data.encode())


def serialize_private_key(private_key):
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def deserialize_private_key(pem_data):
    return serialization.load_pem_private_key(pem_data.encode(), password=None)


def derive_shared_key(private_key, peer_public_key):
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"handshake data",
    ).derive(shared_secret)


# ----------------- Remote Attestation (Helper Functions) -----------------


def run_gpu_attestation(nonce: bytes) -> str:
    """
    Call the NVIDIA Local GPU Verifier (nvtrust) and retrieve the JWT
    Entity Attestation Token (EAT). Assumes the package
        pip install nv-local-gpu-verifier
    is installed.
    """
    out = subprocess.check_output(
        ["python", "-m", "verifier.cc_admin", "--user_mode", "--nonce", nonce.hex()],
        text=True,
    )
    tokens_str = out.split("Entity Attestation Token:", 1)[1].strip()
    platform_token = json.loads(tokens_str)[0][-1]
    gpu_token = json.loads(tokens_str)[1]
    gpu_eat = {"platform_token": platform_token, "gpu_token": gpu_token}
    return json.dumps(gpu_eat)


# ------------------ Attestation ------------------


def create_attestation_report(
    agent_name: str, public_key, nonce: bytes
) -> Tuple[dict, bytes, str]:
    gpu_eat = run_gpu_attestation(nonce)
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key_hash = hashes.Hash(hashes.SHA256())
    public_key_hash.update(public_key_bytes)
    pubkey_digest = public_key_hash.finalize()
    report = {
        "agent_name": agent_name,
        "pubkey_hash": base64.b64encode(pubkey_digest).decode(),
        "gpu_eat_hash": base64.b64encode(
            hashlib.sha256(gpu_eat.encode()).digest()
        ).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "timestamp": time.time(),
    }
    report_json = json.dumps(report, separators=(",", ":")).encode()
    return report, report_json, gpu_eat


def sign_attestation(attestation_json, private_key):
    return base64.b64encode(
        private_key.sign(attestation_json, ec.ECDSA(hashes.SHA256()))
    ).decode()


def verify_attestation(attestation_report, attestation_json, signature_b64, public_key):
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(signature, attestation_json, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False

    pub_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_key_hash = hashes.Hash(hashes.SHA256())
    pub_key_hash.update(pub_key_bytes)
    digest = base64.b64encode(pub_key_hash.finalize()).decode()

    return digest == attestation_report["public_key_hash"]


def verify_attestation_full(
    report_json: bytes,
    signature_b64: str,
    gpu_eat_json: str,
    public_key,
    expected_nonce: bytes,
):
    # 1) signature check
    sig = base64.b64decode(signature_b64)
    try:
        public_key.verify(sig, report_json, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as e:
        raise ValueError("software‑level signature invalid") from e

    report = json.loads(report_json)

    pub_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_key_hash = hashes.Hash(hashes.SHA256())
    pub_key_hash.update(pub_key_bytes)
    digest = base64.b64encode(pub_key_hash.finalize()).decode()

    if report["pubkey_hash"] != digest:
        raise ValueError("pubkey hash mismatch")
    if (
        report["gpu_eat_hash"]
        != base64.b64encode(hashlib.sha256(gpu_eat_json.encode()).digest()).decode()
    ):
        raise ValueError("gpu_eat hash mismatch")
    if report["nonce"] != base64.b64encode(expected_nonce).decode():
        raise ValueError("nonce mismatch (replay?)")

    # 2) GPU evidence check
    platform_jwt, gpu_jwt = (
        json.loads(gpu_eat_json)["platform_token"],
        json.loads(gpu_eat_json)["gpu_token"],
    )
    plat_claims = jwt.decode(platform_jwt, options={"verify_signature": False})

    if plat_claims["eat_nonce"] != expected_nonce.hex():
        raise ValueError("platform nonce mismatch")

    for gpu_idx, gpt_token in gpu_jwt.items():

        gpu_claims = jwt.decode(gpt_token, options={"verify_signature": False})

        if not gpu_claims.get("x-nvidia-gpu-attestation-report-signature-verified"):
            raise ValueError(
                f"GPU attestation signature check failed for GPU {gpu_idx}"
            )

    return True


# ------------------ Secure Messaging ------------------


def encrypt_and_sign(message, key, signing_key, nonce):
    aesgcm = AESGCM(key)
    iv = os.urandom(12)
    ciphertext = aesgcm.encrypt(iv, message.encode(), None)
    data_to_sign = nonce.to_bytes(8, "big") + iv + ciphertext
    signature = signing_key.sign(data_to_sign, ec.ECDSA(hashes.SHA256()))
    return {
        "nonce": nonce,
        "iv": base64.b64encode(iv).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "signature": base64.b64encode(signature).decode(),
    }


def decrypt_and_verify(payload, key, verifying_key):
    nonce = payload["nonce"]
    iv = base64.b64decode(payload["iv"])
    ciphertext = base64.b64decode(payload["ciphertext"])
    signature = base64.b64decode(payload["signature"])
    data_to_verify = nonce.to_bytes(8, "big") + iv + ciphertext

    try:
        verifying_key.verify(signature, data_to_verify, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return "Invalid Signature"

    aesgcm = AESGCM(key)
    decrypted = aesgcm.decrypt(iv, ciphertext, None)
    return decrypted.decode()


# ------------------ Parallel Secure Messaging ------------------

# Constants for parallel processing
CHUNK_SIZE = 16 * 1024  # 16 KiB slices
MAX_WORKERS = min(32, os.cpu_count() or 4)  # Limit max workers


def _iv_from_nonce(nonce: int) -> bytes:
    return nonce.to_bytes(12, "big")[-12:]


def _seal_slice(
    idx: int, block: bytes, shared_key: bytes, nonce: int
) -> Tuple[int, dict]:
    """Encrypt and authenticate a single chunk of data."""
    iv = _iv_from_nonce(nonce)
    aesgcm = AESGCM(shared_key)
    ct = aesgcm.encrypt(iv, block, None)
    tag = hmac.new(shared_key, ct, hashlib.sha256).digest()[:16]
    return idx, {"nonce": nonce, "iv": iv, "ct": ct, "mac": tag}


def _open_slice(idx: int, env: dict, shared_key: bytes) -> Tuple[int, bytes]:
    """Verify and decrypt a single chunk of data."""
    # idx, env = idx_env  # This line is commented out as we now pass idx and env separately
    iv, ct, mac = env["iv"], env["ct"], env["mac"]
    exp = hmac.new(shared_key, ct, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(mac, exp):
        raise ValueError("HMAC mismatch")
    aesgcm = AESGCM(shared_key)
    pt_bytes = aesgcm.decrypt(iv, ct, None)
    return idx, pt_bytes


def _mk_session_header(slices: List[dict]) -> bytes:
    """Create a session header by hashing all slice metadata."""
    h = hashlib.sha256()
    for env in slices:
        h.update(env["nonce"].to_bytes(8, "big"))
        h.update(env["iv"])
        h.update(env["ct"])
        h.update(env["mac"])
    return h.digest()


def encrypt_and_sign_parallel(message: str, key: bytes, signing_key, nonce: int):
    """
    Encrypt and sign a message using sequential processing.

    This function has the same API as encrypt_and_sign but processes large messages
    in chunks for better performance.
    """
    message_bytes = message.encode()

    slices = []
    for idx, off in enumerate(range(0, len(message_bytes), CHUNK_SIZE)):
        block = message_bytes[off : off + CHUNK_SIZE]
        _, env = _seal_slice(idx, block, key, nonce + idx)
        slices.append(env)

    # Create and sign header
    header = _mk_session_header(slices)
    signature = signing_key.sign(header, ec.ECDSA(hashes.SHA256()))

    # Format result to match the original API
    return {
        "nonce": nonce,
        "slices": [
            {
                "nonce": s["nonce"],
                "iv": base64.b64encode(s["iv"]).decode(),
                "ct": base64.b64encode(s["ct"]).decode(),
                "mac": base64.b64encode(s["mac"]).decode(),
            }
            for s in slices
        ],
        "header": base64.b64encode(header).decode(),
        "signature": base64.b64encode(signature).decode(),
    }


def decrypt_and_verify_parallel(payload, key: bytes, verifying_key):
    """
    Decrypt and verify a message using sequential processing.

    This function has the same API as decrypt_and_verify but processes large messages
    in chunks for better performance.
    """

    # Extract and verify header signature
    header = base64.b64decode(payload["header"])
    signature = base64.b64decode(payload["signature"])

    try:
        verifying_key.verify(signature, header, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return "Invalid Signature"

    # Prepare slices for decryption
    slices = [
        {
            "nonce": s["nonce"],
            "iv": base64.b64decode(s["iv"]),
            "ct": base64.b64decode(s["ct"]),
            "mac": base64.b64decode(s["mac"]),
        }
        for s in payload["slices"]
    ]

    parts = [None] * len(slices)
    for idx, env in enumerate(slices):
        _, pt = _open_slice(idx, env, key)
        parts[idx] = pt

    # Combine all decrypted parts
    return b"".join(parts).decode()
