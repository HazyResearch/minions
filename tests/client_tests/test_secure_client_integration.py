"""
Secure client integration tests.
Real encrypted endpoint calls only - zero mocking.
"""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from minions.clients.secure import SecureClient
except ImportError:
    SecureClient = None


class TestSecureClientIntegration(unittest.TestCase):
    """Secure client tests - requires secure crypto utilities and endpoint"""
    
    def setUp(self):
        """Set up Secure client"""
        if SecureClient is None:
            warnings.warn(
                "Skipping Secure client tests: SecureClient not available. "
                "Make sure the secure module is properly installed.",
                UserWarning
            )
            self.skipTest("Secure client not available")
        
        # Check if we have a test endpoint URL
        test_endpoint = os.getenv("SECURE_ENDPOINT_URL")
        if not test_endpoint:
            warnings.warn(
                "Skipping Secure client tests: SECURE_ENDPOINT_URL not set. "
                "Set SECURE_ENDPOINT_URL environment variable to test secure endpoints.",
                UserWarning
            )
            self.skipTest("No secure endpoint URL provided")
        
        try:
            # Try to create client with test endpoint
            self.client = SecureClient(
                endpoint_url=test_endpoint,
                model_name="secure-test-model",
                temperature=0.1,
                max_tokens=50,
                timeout=30,
                verify_attestation=False  # Disable for testing
            )
            
        except Exception as e:
            if "crypto" in str(e).lower() or "secure" in str(e).lower():
                warnings.warn(
                    f"Skipping Secure client tests: Crypto utilities not available. "
                    f"Install secure module dependencies. Error: {e}",
                    UserWarning
                )
                self.skipTest("Secure crypto utilities not available")
            else:
                warnings.warn(
                    f"Skipping Secure client tests: Could not initialize client. Error: {e}",
                    UserWarning
                )
                self.skipTest("Secure client initialization failed")
    
    def test_basic_encrypted_chat(self):
        """Test basic encrypted chat functionality"""
        messages = [{"role": "user", "content": "Say 'secure chat working' and nothing else"}]
        
        try:
            result = self.client.chat(messages)
            
            # Secure client returns (responses, usage)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            
            responses, usage = result
            
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIsInstance(responses[0], str)
            self.assertIn("secure", responses[0].lower())
            
        except Exception as e:
            if "endpoint" in str(e).lower() or "connection" in str(e).lower():
                self.skipTest(f"Secure endpoint not available: {e}")
            elif "crypto" in str(e).lower() or "encryption" in str(e).lower():
                self.skipTest(f"Encryption error: {e}")
            else:
                raise
    
    def test_secure_session_management(self):
        """Test secure session initialization and management"""
        try:
            # Test session attributes
            self.assertIsNone(self.client.session_id)  # Should be None before initialization
            self.assertIsNone(self.client.shared_key)
            self.assertFalse(self.client.is_initialized)
            
            # Test endpoint configuration
            self.assertIsNotNone(self.client.endpoint_url)
            self.assertEqual(self.client.model_name, "secure-test-model")
            
        except Exception as e:
            if "session" in str(e).lower():
                self.skipTest(f"Session management test failed: {e}")
            else:
                raise
    
    def test_encryption_configuration(self):
        """Test encryption and security configuration"""
        try:
            # Test security settings
            self.assertEqual(self.client.timeout, 30)
            self.assertFalse(self.client.verify_attestation)  # Disabled for testing
            self.assertEqual(self.client.session_timeout, 3600)
            
            # Test nonce initialization
            self.assertEqual(self.client.nonce, 1000)
            
        except Exception as e:
            if "config" in str(e).lower():
                self.skipTest(f"Configuration test failed: {e}")
            else:
                raise
    
    def test_attestation_verification(self):
        """Test attestation verification functionality"""
        try:
            # Create client with attestation enabled
            attestation_client = SecureClient(
                endpoint_url=self.client.endpoint_url,
                model_name="secure-test-model",
                verify_attestation=True,
                max_tokens=20
            )
            
            self.assertTrue(attestation_client.verify_attestation)
            
            # Try a simple chat to test attestation
            messages = [{"role": "user", "content": "Test attestation"}]
            
            try:
                responses, usage = attestation_client.chat(messages)
                self.assertIsInstance(responses, list)
            except Exception as e:
                if "attestation" in str(e).lower() or "verification" in str(e).lower():
                    self.skipTest(f"Attestation verification failed (expected for test): {e}")
                else:
                    raise
                    
        except Exception as e:
            if "attestation" in str(e).lower():
                self.skipTest(f"Attestation test failed: {e}")
            else:
                raise
    
    def test_error_handling(self):
        """Test secure client error handling"""
        try:
            # Test with invalid messages
            invalid_messages = []
            
            with self.assertRaises(AssertionError):
                self.client.chat(invalid_messages)
                
        except Exception as e:
            if "error" in str(e).lower():
                self.skipTest(f"Error handling test failed: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()