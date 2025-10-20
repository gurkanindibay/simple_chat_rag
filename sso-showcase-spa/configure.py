#!/usr/bin/env python3
"""
Configuration helper for SSO Showcase SPA
Updates the index.html file with the actual Client ID
"""
import sys
import re
from pathlib import Path

def update_client_id(client_id: str):
    """Update the Client ID in index.html"""
    index_file = Path(__file__).parent / "index.html"
    
    if not index_file.exists():
        print(f"❌ Error: {index_file} not found")
        return False
    
    # Read the file
    content = index_file.read_text()
    
    # Replace the placeholder
    updated_content = content.replace(
        "clientId: 'SSO_SHOWCASE_CLIENT_ID'",
        f"clientId: '{client_id}'"
    )
    
    if content == updated_content:
        print("⚠️  Warning: Client ID placeholder not found or already configured")
        print("    Current configuration:")
        # Extract current client ID
        match = re.search(r"clientId:\s*['\"]([^'\"]+)['\"]", content)
        if match:
            print(f"    Client ID: {match.group(1)}")
        return False
    
    # Write back
    index_file.write_text(updated_content)
    print(f"✅ Successfully updated Client ID to: {client_id}")
    print(f"📝 Updated file: {index_file}")
    return True

def show_current_config():
    """Show current configuration"""
    index_file = Path(__file__).parent / "index.html"
    
    if not index_file.exists():
        print(f"❌ Error: {index_file} not found")
        return
    
    content = index_file.read_text()
    
    # Extract configuration
    client_id_match = re.search(r"clientId:\s*['\"]([^'\"]+)['\"]", content)
    redirect_uri_match = re.search(r"redirectUri:\s*['\"]([^'\"]+)['\"]", content)
    authority_match = re.search(r"authority:\s*['\"]([^'\"]+)['\"]", content)
    
    print("\n📋 Current SSO Showcase SPA Configuration:")
    print("=" * 60)
    
    if client_id_match:
        client_id = client_id_match.group(1)
        if client_id == "SSO_SHOWCASE_CLIENT_ID":
            print("❌ Client ID: NOT CONFIGURED (placeholder)")
        else:
            print(f"✅ Client ID: {client_id}")
    
    if redirect_uri_match:
        print(f"✅ Redirect URI: {redirect_uri_match.group(1)}")
    
    if authority_match:
        print(f"✅ Authority: {authority_match.group(1)}")
    
    print("=" * 60)

def main():
    if len(sys.argv) < 2:
        print("SSO Showcase SPA Configuration Helper")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 configure.py <client-id>     - Set the Client ID")
        print("  python3 configure.py --show          - Show current config")
        print("\nExample:")
        print("  python3 configure.py abcd1234-5678-90ef-ghij-klmnopqrstuv")
        print("")
        show_current_config()
        return
    
    if sys.argv[1] == "--show":
        show_current_config()
        return
    
    client_id = sys.argv[1].strip()
    
    # Basic validation
    if len(client_id) < 30:
        print("⚠️  Warning: Client ID seems too short. Azure Client IDs are typically 36 characters.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    if update_client_id(client_id):
        print("\n🎉 Configuration complete!")
        print("\nNext steps:")
        print("1. Ensure this Client ID is registered in Azure Portal")
        print("2. Add http://localhost:8001 as a redirect URI (SPA platform)")
        print("3. Run: python3 serve.py")
        print("4. Open http://localhost:8001 in your browser")

if __name__ == "__main__":
    main()
