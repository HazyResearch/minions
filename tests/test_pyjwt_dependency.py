"""
Test PyJWT dependency resolution in setup.py.

This test ensures that:
1. Base installation includes PyJWT
2. Secure installation includes PyJWT[crypto] with cryptography support
3. No duplicate PyJWT packages are installed
4. JWT functionality works in both installations
"""

import subprocess
import sys
import json
import tempfile
import venv
from pathlib import Path


def test_current_installation_has_jwt():
    """Test that JWT can be imported in current environment."""
    try:
        import jwt
        print(f"✓ Current environment: jwt version {jwt.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Current environment: JWT import failed: {e}")
        return False


def test_jwt_basic_functionality():
    """Test basic JWT encoding/decoding (HS256 - no crypto needed)."""
    import jwt

    payload = {"user": "test", "data": "example"}
    secret = "test_secret"

    # Encode
    token = jwt.encode(payload, secret, algorithm="HS256")
    assert isinstance(token, str), "Token should be a string"
    print(f"✓ JWT encoding works (HS256)")

    # Decode
    decoded = jwt.decode(token, secret, algorithms=["HS256"])
    assert decoded["user"] == "test", "Decoded payload should match"
    assert decoded["data"] == "example", "Decoded data should match"
    print(f"✓ JWT decoding works (HS256)")

    return True


def test_cryptography_available():
    """Test that cryptography library is available (needed for RS256/ES384)."""
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes

        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        print("✓ Cryptography library available")
        return True
    except ImportError as e:
        print(f"✗ Cryptography library not available: {e}")
        return False


def test_jwt_crypto_algorithms():
    """Test JWT with RSA algorithms (requires PyJWT[crypto])."""
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
        payload = {"user": "test", "secure": True}
        token = jwt.encode(payload, private_key, algorithm="RS256")
        assert isinstance(token, str), "RS256 token should be a string"
        print("✓ JWT RS256 encoding works")

        # Decode with RS256
        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        assert decoded["user"] == "test", "Decoded payload should match"
        assert decoded["secure"] is True, "Decoded secure flag should match"
        print("✓ JWT RS256 decoding works")

        return True
    except ImportError as e:
        print(f"✗ JWT crypto algorithms not available: {e}")
        return False
    except Exception as e:
        print(f"✗ JWT crypto test failed: {e}")
        return False


def test_no_duplicate_pyjwt_packages():
    """Test that PyJWT is not installed twice."""
    result = subprocess.run(
        ["pip", "list", "--format=json"],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print(f"✗ Failed to list packages: {result.stderr}")
        return False

    packages = json.loads(result.stdout)
    pyjwt_packages = [
        p for p in packages
        if p['name'].lower() in ['pyjwt', 'jwt']
    ]

    if len(pyjwt_packages) == 0:
        print("✗ PyJWT not found in installed packages")
        return False
    elif len(pyjwt_packages) == 1:
        pkg = pyjwt_packages[0]
        print(f"✓ Single PyJWT package: {pkg['name']} version {pkg['version']}")
        return True
    else:
        print(f"✗ Multiple PyJWT packages found: {pyjwt_packages}")
        return False


def test_setup_py_has_correct_pyjwt():
    """Test that setup.py uses official PyJWT name (not lowercase pyjwt)."""
    setup_path = Path(__file__).parent.parent / "setup.py"

    if not setup_path.exists():
        print(f"✗ setup.py not found at {setup_path}")
        return False

    content = setup_path.read_text()

    # Check for lowercase "pyjwt" in install_requires or extras_require
    lines = content.split('\n')
    issues = []

    for i, line in enumerate(lines, 1):
        if '"pyjwt"' in line.lower():
            # Check if it's exactly "pyjwt" (not "PyJWT")
            if '"pyjwt"' in line and '"PyJWT' not in line:
                issues.append(f"Line {i}: Found lowercase 'pyjwt': {line.strip()}")

    if issues:
        print("✗ setup.py has lowercase 'pyjwt' entries:")
        for issue in issues:
            print(f"  {issue}")
        return False

    # Check for PyJWT in install_requires
    has_pyjwt_base = '"PyJWT"' in content
    has_pyjwt_crypto = '"PyJWT[crypto]"' in content

    if has_pyjwt_base:
        print("✓ setup.py uses 'PyJWT' in base requirements")
    else:
        print("✗ setup.py missing 'PyJWT' in base requirements")

    if has_pyjwt_crypto:
        print("✓ setup.py uses 'PyJWT[crypto]' in secure extras")
    else:
        print("✗ setup.py missing 'PyJWT[crypto]' in secure extras")

    return has_pyjwt_base and has_pyjwt_crypto and len(issues) == 0


def main():
    """Run all PyJWT dependency tests."""
    print("\n" + "="*60)
    print("PyJWT Dependency Tests")
    print("="*60 + "\n")

    tests = [
        ("Current Installation Has JWT", test_current_installation_has_jwt),
        ("JWT Basic Functionality (HS256)", test_jwt_basic_functionality),
        ("Cryptography Available", test_cryptography_available),
        ("JWT Crypto Algorithms (RS256)", test_jwt_crypto_algorithms),
        ("No Duplicate PyJWT Packages", test_no_duplicate_pyjwt_packages),
        ("setup.py Has Correct PyJWT", test_setup_py_has_correct_pyjwt),
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
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
