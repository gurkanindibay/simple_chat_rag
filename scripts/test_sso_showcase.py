#!/usr/bin/env python3
"""
Test script for SSO Showcase functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.sso_showcase import generate_sso_showcase_page

def test_sso_showcase_generation():
    """Test that the SSO showcase page generates correctly"""
    
    # Mock user info as it would come from a JWT token
    test_user_info = {
        "user_id": "12345678-1234-1234-1234-123456789abc",
        "email": "john.doe@example.com",
        "name": "John Doe",
        "tenant_id": "87654321-4321-4321-4321-cba987654321",
        "app_id": "test-app-id",
        "roles": ["rag_chat_user", "admin"],
        "scopes": ["access_as_user", "User.Read"]
    }
    
    # Generate the HTML page
    html_output = generate_sso_showcase_page(test_user_info)
    
    # Basic validation
    assert "John Doe" in html_output, "User name should be in output"
    assert "john.doe@example.com" in html_output, "Email should be in output"
    assert "rag_chat_user" in html_output, "Role should be in output"
    assert "access_as_user" in html_output, "Scope should be in output"
    assert "Microsoft Entra ID" in html_output, "SSO provider should be mentioned"
    assert "<!DOCTYPE html>" in html_output, "Should be valid HTML"
    
    print("✅ SSO Showcase page generation test passed!")
    print(f"📄 Generated HTML length: {len(html_output)} characters")
    
    return True


def test_empty_roles_and_scopes():
    """Test page generation with no roles or scopes"""
    
    test_user_info = {
        "user_id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "tenant_id": "test-tenant-id",
        "app_id": "test-app-id",
        "roles": [],
        "scopes": []
    }
    
    html_output = generate_sso_showcase_page(test_user_info)
    
    assert "No roles assigned" in html_output, "Should show message for empty roles"
    assert "No scopes" in html_output, "Should show message for empty scopes"
    
    print("✅ Empty roles/scopes test passed!")
    
    return True


def test_missing_fields():
    """Test page generation with missing optional fields"""
    
    # Minimal user info
    test_user_info = {
        "user_id": "test-user",
        "tenant_id": "test-tenant"
    }
    
    html_output = generate_sso_showcase_page(test_user_info)
    
    # Should handle missing fields gracefully
    assert "N/A" in html_output, "Should show N/A for missing fields"
    assert "<!DOCTYPE html>" in html_output, "Should still generate valid HTML"
    
    print("✅ Missing fields test passed!")
    
    return True


if __name__ == "__main__":
    print("🧪 Testing SSO Showcase Module\n")
    
    try:
        test_sso_showcase_generation()
        test_empty_roles_and_scopes()
        test_missing_fields()
        
        print("\n" + "="*50)
        print("✨ All tests passed successfully!")
        print("="*50)
        print("\n📋 Next steps:")
        print("1. Start the backend server:")
        print("   uvicorn backend.main:app --reload")
        print("\n2. Start the frontend:")
        print("   cd frontend && npm run dev")
        print("\n3. Sign in to the application")
        print("\n4. Click 'SSO Showcase' button in the sidebar")
        print("   or visit: http://localhost:8000/sso")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
