#!/usr/bin/env python3
"""
OAuth2 client management tool for A2A-Minions server.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from a2a_minions.auth import OAuth2ClientManager


def list_clients(manager: OAuth2ClientManager):
    """List all OAuth2 clients."""
    clients = manager.list_clients()
    
    if not clients:
        print("No OAuth2 clients registered.")
        return
    
    print(f"\n{'Client ID':<30} {'Name':<20} {'Active':<8} {'Scopes':<40}")
    print("-" * 100)
    
    for client in clients:
        client_id = client['client_id']
        name = client['name'][:20]
        active = "Yes" if client['active'] else "No"
        scopes = ", ".join(client['scopes'])[:40]
        
        print(f"{client_id:<30} {name:<20} {active:<8} {scopes:<40}")
    
    print(f"\nTotal clients: {len(clients)}")


def register_client(manager: OAuth2ClientManager, name: str, scopes: List[str]):
    """Register a new OAuth2 client."""
    client_id, client_secret = manager.register_client(name, scopes)
    
    print(f"\n✅ OAuth2 client registered successfully!")
    print(f"\nClient ID:     {client_id}")
    print(f"Client Secret: {client_secret}")
    print(f"Name:          {name}")
    print(f"Scopes:        {', '.join(scopes)}")
    print(f"\n⚠️  Save these credentials securely - the secret won't be shown again!")


def revoke_client(manager: OAuth2ClientManager, client_id: str):
    """Revoke an OAuth2 client."""
    if manager.revoke_client(client_id):
        print(f"✅ OAuth2 client revoked: {client_id}")
    else:
        print(f"❌ OAuth2 client not found: {client_id}")


def export_clients(manager: OAuth2ClientManager, output_file: str):
    """Export OAuth2 clients to file (without secrets)."""
    clients = manager.list_clients()
    
    with open(output_file, 'w') as f:
        json.dump(clients, f, indent=2)
    
    print(f"✅ Exported {len(clients)} clients to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Manage OAuth2 clients for A2A-Minions server")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all OAuth2 clients')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new OAuth2 client')
    register_parser.add_argument('name', help='Client name')
    register_parser.add_argument('--scopes', nargs='+', 
                                default=['minion:query', 'minions:query', 'tasks:read', 'tasks:write'],
                                help='Scopes to grant (default: all scopes)')
    
    # Revoke command
    revoke_parser = subparsers.add_parser('revoke', help='Revoke an OAuth2 client')
    revoke_parser.add_argument('client_id', help='Client ID to revoke')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export OAuth2 clients (without secrets)')
    export_parser.add_argument('--output', '-o', default='oauth2_clients_export.json',
                              help='Output file (default: oauth2_clients_export.json)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize OAuth2 client manager
    manager = OAuth2ClientManager()
    
    # Execute command
    if args.command == 'list':
        list_clients(manager)
    
    elif args.command == 'register':
        register_client(manager, args.name, args.scopes)
    
    elif args.command == 'revoke':
        revoke_client(manager, args.client_id)
    
    elif args.command == 'export':
        export_clients(manager, args.output)


if __name__ == '__main__':
    main()