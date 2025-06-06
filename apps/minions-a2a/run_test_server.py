#!/usr/bin/env python3
"""
Run the A2A-Minions server with a test API key for testing.
This script configures the server with the test key "abcd" for easy testing.
"""

import subprocess
import sys

# Run the server with the test API key
subprocess.run([
    sys.executable,
    "run_server.py",
    "--api-key", "abcd",
    "--host", "0.0.0.0",
    "--port", "8001"
])