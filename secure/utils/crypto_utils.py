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
from cryptography import x509
from cryptography.hazmat.primitives import (
    serialization as crypto_serial,
    hashes as crypto_hashes,
)
import hashlib, base64, ssl, socket
from nv_attestation_sdk.attestation import Attestation, Devices, Environment
import base64, json, requests, jwt
from jwt import PyJWKClient, get_unverified_header
from cryptography import x509
from jwt.algorithms import ECAlgorithm

JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
_jwks = PyJWKClient(JWKS_URL)        # caches keys after the first HTTP hit




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

def _jwks_fallback(token: str, alg: str) -> dict:
    """
    Last-chance attempt: download the JWKS and try every key until one
    verifies the signature.  Works even when there's no `kid` or `x5c`.
    """
    jwks = _jwks.fetch_data()        # public method → {"keys":[…]}
    for jwk_dict in jwks["keys"]:
        pubkey = ECAlgorithm.from_jwk(json.dumps(jwk_dict))
        try:
            return jwt.decode(token, key=pubkey, algorithms=[alg], options={"verify_aud": False})
        except jwt.InvalidSignatureError:
            continue
    raise jwt.InvalidSignatureError("None of the JWKS keys matched this signature.")


def decode_gpu_eat(token: str) -> dict:
    """
    Verify & decode a NVIDIA GPU-EAT that may lack `kid`.
    Returns the claim set or raises jwt.InvalidSignatureError.
    """
    hdr = get_unverified_header(token)
    alg = hdr["alg"]                 # ES384 for H100/L40S

    # 1️⃣  Normal path — `kid` is present.
    if "kid" in hdr:
        key = _jwks.get_signing_key_from_jwt(token).key
        return jwt.decode(token, key=key, algorithms=[alg], options={"verify_aud": False})

    # 2️⃣  Self-contained path — use the x5c certificate chain.
    if "x5c" in hdr:
        leaf_der = base64.b64decode(hdr["x5c"][0])
        leaf_pub = x509.load_der_x509_certificate(leaf_der).public_key()
        return jwt.decode(token, key=leaf_pub, algorithms=[alg], options={"verify_aud": False})

    # 3️⃣  Fallback — brute-check every JWKS key.
    return _jwks_fallback(token, alg)


def get_tls_pubkey_hash(cert_path: str) -> str:
    # load the PEM cert from disk
    cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
    # grab the pubkey in DER form
    pubkey_der = cert.public_key().public_bytes(
        crypto_serial.Encoding.DER,
        crypto_serial.PublicFormat.SubjectPublicKeyInfo,
    )
    # sha256 it
    h = crypto_hashes.Hash(crypto_hashes.SHA256())
    h.update(pubkey_der)
    digest = h.finalize()
    # return a base64 string
    return base64.b64encode(digest).decode()


def run_gpu_attestation(nonce: bytes) -> str:
    """
    Call the NVIDIA Local GPU Verifier (nvtrust) and retrieve the JWT
    Entity Attestation Token (EAT). Assumes the package
        pip install nv-local-gpu-verifier
    is installed.
    """
    client = Attestation()
    client.set_name("myNode")
    client.set_nonce(nonce.hex())
    client.add_verifier(
        Devices.GPU,
        Environment.REMOTE,
        "",
        "",
    )
    evidence_list = client.get_evidence()
    ok = client.attest(evidence_list)
    attesation_jwt = client.get_token()
    tokens = json.loads(attesation_jwt)
    gpu_eat = {
        "platform_token": tokens[0][-1],
        "gpu_token": tokens[1]["REMOTE_GPU_CLAIMS"][1],
    }
    return json.dumps(gpu_eat)

def generate_attestation_keys(attestation_key_path: str):
    # check if keys exist
    if os.path.exists(f"{attestation_key_path}/attestation_private.pem") and os.path.exists(f"{attestation_key_path}/attestation_public.pem"):
        # load keys
        with open(f"{attestation_key_path}/attestation_private.pem", "r") as f:
            private_key = deserialize_private_key(f.read())
        with open(f"{attestation_key_path}/attestation_public.pem", "r") as f:
            public_key = deserialize_public_key(f.read())
        return private_key, public_key
    else:
        # generate keys
        # create directory if it doesn't exist
        if not os.path.exists(attestation_key_path):
            os.makedirs(attestation_key_path)

        private_key, public_key = generate_key_pair()
        # save keys
        with open(f"{attestation_key_path}/attestation_private.pem", "w") as f:
            f.write(serialize_private_key(private_key))
        with open(f"{attestation_key_path}/attestation_public.pem", "w") as f:
            f.write(serialize_public_key(public_key))

        return private_key, public_key



def generate_attestation_keys(attestation_key_path: str):
    # check if keys exist
    if os.path.exists(
        f"{attestation_key_path}/attestation_private.pem"
    ) and os.path.exists(f"{attestation_key_path}/attestation_public.pem"):
        # load keys
        with open(f"{attestation_key_path}/attestation_private.pem", "r") as f:
            private_key = deserialize_private_key(f.read())
        with open(f"{attestation_key_path}/attestation_public.pem", "r") as f:
            public_key = deserialize_public_key(f.read())
        return private_key, public_key
    else:
        # generate keys
        # create directory if it doesn't exist
        if not os.path.exists(attestation_key_path):
            os.makedirs(attestation_key_path)

        private_key, public_key = generate_key_pair()
        # save keys
        with open(f"{attestation_key_path}/attestation_private.pem", "w") as f:
            f.write(serialize_private_key(private_key))
        with open(f"{attestation_key_path}/attestation_public.pem", "w") as f:
            f.write(serialize_public_key(public_key))

        return private_key, public_key


# ------------------ Attestation ------------------

# write a function that reads a pem file and returns the hash

def get_pem_hash(pem_file: str) -> str:
    with open(pem_file, "r") as f:
        pem_data = f.read()
    return get_public_key_hash(deserialize_public_key(pem_data))


def get_public_key_hash(public_key) -> str:
    """
    Return the base64-encoded SHA256 hash of an ECDSA public key in DER format.
    :param public_key: A cryptography public key object
    :return: Base64 string of the SHA256 hash
    """
    # Serialize the key to DER format
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key_hash = hashes.Hash(hashes.SHA256())
    public_key_hash.update(public_key_bytes)
    pubkey_digest = public_key_hash.finalize()
    # Return base64-encoded string
    return base64.b64encode(pubkey_digest).decode()

def run_snp_attestation() -> str:
    import subprocess
    # preparation steps in here https://github.com/Azure/confidential-computing-cvm-guest-attestation/tree/main/cvm-attestation-sample-app#build-instructions-for-linux-using-self-contained-attestation-lib
    # including installing and building dependencies
    # This will prompt for the user's sudo password
    result = subprocess.run(["sudo", "/home/azureuser/confidential-computing-cvm-guest-attestation/cvm-attestation-sample-app/AttestationClient", "-o", "token"],
                        capture_output=True,
                        text=True)
    if result.returncode == 0:
        token = result.stdout.strip()
        return token
    else:
        raise Exception(f"Failed to run SNP attestation: {result.stderr}")

def create_attestation_report(
    agent_name: str, public_key, nonce: bytes, tls_cert_path: str
) -> Tuple[dict, bytes, str]:
    gpu_eat = run_gpu_attestation(nonce)
    snp_attestation_token = run_snp_attestation() # DB: added SNP report using https://github.com/Azure/confidential-computing-cvm-guest-attestation/tree/main/cvm-attestation-sample-app#build-instructions-for-linux-using-self-contained-attestation-lib
    tls_pubkey_hash = get_tls_pubkey_hash(tls_cert_path)
    public_key_hash = get_public_key_hash(public_key)
    
  
    report = {
        "agent_name": agent_name,
        "pubkey_hash": public_key_hash,
        "snp_attestation_token": snp_attestation_token,
        "gpu_eat_hash": base64.b64encode(
            hashlib.sha256(gpu_eat.encode()).digest()
        ).decode(),
        "tls_pubkey_hash": tls_pubkey_hash,
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


def get_server_tls_pubkey_hash(host: str, port: int = 443) -> str:

    # open a TLS connection and pull the raw DER cert
    ctx = ssl.create_default_context()
    with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
        s.connect((host, port))
        der_cert = s.getpeercert(binary_form=True)

    # load it into cryptography to extract the pubkey
    from cryptography import x509

    cert = x509.load_der_x509_certificate(der_cert)
    pubkey_der = cert.public_key().public_bytes(
        crypto_serial.Encoding.DER,
        crypto_serial.PublicFormat.SubjectPublicKeyInfo,
    )

    # hash and b64
    digest = hashlib.sha256(pubkey_der).digest()
    return base64.b64encode(digest).decode()


def verify_attestation_full(
    report_json: bytes,
    signature_b64: str,
    gpu_eat_json: str,
    public_key,
    expected_nonce: bytes,
    server_host: str,
    server_port: int = 443,
    trusted_attestation_hash: str = None,
):
    # 1) signature check
    sig = base64.b64decode(signature_b64)
    try:
        public_key.verify(sig, report_json, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as e:
        raise ValueError("software‑level signature invalid") from e

    report = json.loads(report_json)
    seen_hash = get_server_tls_pubkey_hash(server_host, server_port)
    if report["tls_pubkey_hash"] != seen_hash:
        raise ValueError("TLS pubkey hash mismatch")


    # 2) Pin the signer key
    pub_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_key_hash = hashes.Hash(hashes.SHA256())
    pub_key_hash.update(pub_key_bytes)
    digest = base64.b64encode(pub_key_hash.finalize()).decode()

    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_hash = base64.b64encode(hashlib.sha256(pub_bytes).digest()).decode()
    if key_hash != trusted_attestation_hash:
        raise ValueError("Attestation signer key is not trusted") 
    

    # 3) check hashes

    if report["pubkey_hash"] != digest:
        raise ValueError("pubkey hash mismatch")
    if (
        report["gpu_eat_hash"]
        != base64.b64encode(hashlib.sha256(gpu_eat_json.encode()).digest()).decode()
    ):
        raise ValueError("gpu_eat hash mismatch")
    if report["nonce"] != base64.b64encode(expected_nonce).decode():
        raise ValueError("nonce mismatch (replay?)")
    
    # 5) SNP attestation check
    snp_attestation_token = report["snp_attestation_token"]
    # DB: add SNP attestation check here

    # 5) GPU evidence check
    platform_jwt, gpu_jwt = (
        json.loads(gpu_eat_json)["platform_token"],
        json.loads(gpu_eat_json)["gpu_token"],
    )

    for gpu_idx, gpt_token in gpu_jwt.items():
        #gpu_claims = jwt.decode(gpt_token, options={"verify_signature": False})
        gpu_claims = decode_gpu_eat(gpt_token)
        if gpu_claims.get("eat_nonce") != expected_nonce.hex():
            raise ValueError("platform nonce mismatch")

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
    try:
        # Check if this is a parallel format payload (has "slices" and "header")
        if "slices" in payload and "header" in payload:
            # Use parallel decryption
            return decrypt_and_verify_parallel(payload, key, verifying_key)

        # Regular format
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
    except Exception as e:
        # Log detailed error information for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"decrypt_and_verify failed: {type(e).__name__}: {str(e)}")
        logger.error(
            f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}"
        )
        if isinstance(payload, dict):
            logger.error(f"Nonce: {payload.get('nonce', 'Missing')}")
            if "iv" in payload:
                try:
                    logger.error(f"IV length: {len(base64.b64decode(payload['iv']))}")
                except:
                    logger.error("IV: Invalid base64")
            else:
                logger.error("IV: Missing")
            if "ciphertext" in payload:
                try:
                    logger.error(
                        f"Ciphertext length: {len(base64.b64decode(payload['ciphertext']))}"
                    )
                except:
                    logger.error("Ciphertext: Invalid base64")
            else:
                logger.error("Ciphertext: Missing")
        raise


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
