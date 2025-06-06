#!/usr/bin/env python3
"""
CLI tool for managing A2A-Minions API keys.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from a2a_minions.auth import AuthConfig, APIKeyManager


def list_keys(api_key_manager: APIKeyManager):
    """List all API keys."""
    if not api_key_manager.api_keys:
        print("No API keys found.")
        return
    
    # Prepare data for tabulation
    headers = ["Key (last 8 chars)", "Name", "Created", "Active", "Scopes"]
    rows = []
    
    for key, data in api_key_manager.api_keys.items():
        # Show only last 8 characters of key for security
        key_display = f"...{key[-8:]}"
        created = data.get("created_at", "Unknown")
        if created != "Unknown":
            # Format datetime
            try:
                dt = datetime.fromisoformat(created)
                created = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        active = "✓" if data.get("active", True) else "✗"
        scopes = ", ".join(data.get("scopes", []))
        
        rows.append([key_display, data.get("name", "Unknown"), created, active, scopes])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def generate_key(api_key_manager: APIKeyManager, name: str, scopes: list):
    """Generate a new API key."""
    key = api_key_manager.generate_api_key(name, scopes)
    print(f"Generated new API key for '{name}':")
    print(f"\n{key}\n")
    print("⚠️  Save this key securely - it won't be shown again!")
    print(f"Scopes: {', '.join(scopes)}")


def revoke_key(api_key_manager: APIKeyManager, key_pattern: str):
    """Revoke an API key."""
    # Find keys that match the pattern (last 8 chars)
    matching_keys = []
    for key in api_key_manager.api_keys:
        if key.endswith(key_pattern) or key == key_pattern:
            matching_keys.append(key)
    
    if not matching_keys:
        print(f"No keys found matching pattern: {key_pattern}")
        return
    
    if len(matching_keys) > 1:
        print(f"Multiple keys match pattern '{key_pattern}':")
        for key in matching_keys:
            print(f"  - ...{key[-8:]}")
        print("Please provide a more specific pattern.")
        return
    
    key = matching_keys[0]
    if api_key_manager.revoke_api_key(key):
        print(f"Revoked API key: ...{key[-8:]}")
    else:
        print(f"Failed to revoke key.")


def export_keys(api_key_manager: APIKeyManager, output_file: str):
    """Export API keys to a file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(api_key_manager.api_keys, f, indent=2)
        print(f"Exported {len(api_key_manager.api_keys)} keys to {output_file}")
    except Exception as e:
        print(f"Error exporting keys: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage A2A-Minions API keys")
    parser.add_argument("--api-keys-file", default="api_keys.json", help="Path to API keys file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all API keys")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a new API key")
    gen_parser.add_argument("name", help="Name for the API key")
    gen_parser.add_argument("--scopes", nargs="+", 
                           default=["minion:query", "minions:query", "tasks:read", "tasks:write"],
                           help="Scopes for the API key")
    
    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("key", help="API key or last 8 characters")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export API keys")
    export_parser.add_argument("output", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize API key manager
    config = AuthConfig(api_keys_file=args.api_keys_file)
    api_key_manager = APIKeyManager(config)
    
    # Execute command
    if args.command == "list":
        list_keys(api_key_manager)
    elif args.command == "generate":
        generate_key(api_key_manager, args.name, args.scopes)
    elif args.command == "revoke":
        revoke_key(api_key_manager, args.key)
    elif args.command == "export":
        export_keys(api_key_manager, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()