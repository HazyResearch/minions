"""
Integration test for JWT usage across the minions codebase.

This ensures that:
1. secure/utils/crypto_utils.py can use JWT with crypto algorithms (RS256, ES384)
2. apps/minions-a2a/a2a_minions/auth.py can use JWT with HS256
3. No import errors after PyJWT dependency change
"""

import sys
from pathlib import Path


def test_secure_crypto_utils_imports():
    """Test that secure crypto utils can import jwt and related modules."""
    try:
        # This import path matches the actual usage in secure/utils/crypto_utils.py
        import jwt
        from jwt import PyJWKClient, get_unverified_header
        from jwt.algorithms import ECAlgorithm
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes

        print("✓ secure/utils/crypto_utils.py imports work")
        return True
    except ImportError as e:
        print(f"✗ secure/utils/crypto_utils.py imports failed: {e}")
        return False


def test_a2a_auth_imports():
    """Test that a2a auth module can import jwt."""
    try:
        import jwt
        print("✓ apps/minions-a2a/a2a_minions/auth.py imports work")
        return True
    except ImportError as e:
        print(f"✗ apps/minions-a2a/a2a_minions/auth.py imports failed: {e}")
        return False


def test_jwt_es384_algorithm():
    """Test ES384 algorithm used in secure/utils/crypto_utils.py decode_gpu_eat()."""
    try:
        import jwt
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.backends import default_backend

        # Generate EC key pair (ES384 uses P-384 curve)
        private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
        public_key = private_key.public_key()

        # Encode with ES384
        payload = {"test": "gpu_eat", "x-nvidia-gpu-id": "test-gpu"}
        token = jwt.encode(payload, private_key, algorithm="ES384")
        assert isinstance(token, str), "ES384 token should be a string"

        # Decode with ES384
        decoded = jwt.decode(token, public_key, algorithms=["ES384"])
        assert decoded["test"] == "gpu_eat", "Decoded payload should match"

        print("✓ ES384 algorithm works (used by decode_gpu_eat)")
        return True
    except Exception as e:
        print(f"✗ ES384 algorithm failed: {e}")
        return False


def test_jwt_rs256_algorithm():
    """Test RS256 algorithm used in secure/utils/crypto_utils.py verify_azure_attestation_token()."""
    try:
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Encode with RS256
        payload = {"x-ms-attestation-type": "azurevm", "secureboot": True}
        token = jwt.encode(payload, private_key, algorithm="RS256")
        assert isinstance(token, str), "RS256 token should be a string"

        # Decode with RS256
        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        assert decoded["x-ms-attestation-type"] == "azurevm", "Decoded payload should match"

        print("✓ RS256 algorithm works (used by verify_azure_attestation_token)")
        return True
    except Exception as e:
        print(f"✗ RS256 algorithm failed: {e}")
        return False


def test_jwt_hs256_algorithm():
    """Test HS256 algorithm used in apps/minions-a2a/a2a_minions/auth.py."""
    try:
        import jwt
        from datetime import datetime, timedelta

        # Generate token with HS256 (like JWTManager.create_token)
        secret = "test_secret_key"
        payload = {
            "sub": "test_client",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "scopes": ["tasks:read", "tasks:write"]
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        assert isinstance(token, str), "HS256 token should be a string"

        # Decode with HS256 (like JWTManager.verify_token)
        decoded = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_exp": False}
        )
        assert decoded["sub"] == "test_client", "Decoded payload should match"

        print("✓ HS256 algorithm works (used by a2a auth)")
        return True
    except Exception as e:
        print(f"✗ HS256 algorithm failed: {e}")
        return False


def test_pyjwk_client():
    """Test PyJWKClient used in secure/utils/crypto_utils.py."""
    try:
        from jwt import PyJWKClient

        # Don't actually fetch JWKS (network call), just verify import
        # and class is available
        assert PyJWKClient is not None, "PyJWKClient should be available"

        print("✓ PyJWKClient available (used by decode_gpu_eat)")
        return True
    except Exception as e:
        print(f"✗ PyJWKClient import failed: {e}")
        return False


def test_jwt_exception_handling():
    """Test JWT exception classes used in auth.py."""
    try:
        import jwt

        # Test exception classes
        assert hasattr(jwt, 'ExpiredSignatureError'), "ExpiredSignatureError should exist"
        assert hasattr(jwt, 'InvalidTokenError'), "InvalidTokenError should exist"
        assert hasattr(jwt, 'InvalidSignatureError'), "InvalidSignatureError should exist"

        # Test they can be caught
        secret = "test"
        token = jwt.encode({"exp": 0}, secret, algorithm="HS256")

        try:
            jwt.decode(token, secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            pass  # Expected

        print("✓ JWT exception classes work (used by auth.py)")
        return True
    except Exception as e:
        print(f"✗ JWT exception handling failed: {e}")
        return False


def main():
    """Run JWT integration tests."""
    print("\n" + "="*60)
    print("JWT Integration Tests")
    print("="*60 + "\n")

    tests = [
        ("Secure crypto_utils imports", test_secure_crypto_utils_imports),
        ("A2A auth imports", test_a2a_auth_imports),
        ("ES384 algorithm (GPU attestation)", test_jwt_es384_algorithm),
        ("RS256 algorithm (Azure attestation)", test_jwt_rs256_algorithm),
        ("HS256 algorithm (A2A auth)", test_jwt_hs256_algorithm),
        ("PyJWKClient (JWKS)", test_pyjwk_client),
        ("JWT exception handling", test_jwt_exception_handling),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All JWT integration tests passed!")
        print("✓ PyJWT dependency change is safe!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
