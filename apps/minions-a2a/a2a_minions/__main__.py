"""
Entry point for running A2A-Minions as a module.

Usage:
    python -m a2a_minions
    python -m a2a_minions.server
"""

from .server import main

if __name__ == "__main__":
    main() 