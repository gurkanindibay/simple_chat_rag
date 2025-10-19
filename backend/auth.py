"""
Microsoft Entra (Azure AD) Authentication Module
Handles JWT token validation and user authentication
"""
import os
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
import requests
from functools import lru_cache

security = HTTPBearer()


class EntraAuthConfig:
    """Configuration for Microsoft Entra authentication"""
    
    def __init__(self):
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")  # Optional, for backend-only flows
        
        if not self.tenant_id or not self.client_id:
            raise ValueError(
                "Missing required environment variables: AZURE_TENANT_ID and AZURE_CLIENT_ID"
            )
        
        # Construct the well-known OpenID configuration URL
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"
        self.jwks_uri = f"{self.authority}/.well-known/openid-configuration"
        
    @property
    def issuer(self):
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


@lru_cache()
def get_auth_config() -> EntraAuthConfig:
    """Get cached authentication configuration"""
    return EntraAuthConfig()


@lru_cache()
def get_jwks_client() -> PyJWKClient:
    """Get cached JWKS client for token validation"""
    config = get_auth_config()
    
    # Fetch OpenID configuration to get JWKS URI
    response = requests.get(f"{config.authority}/.well-known/openid-configuration")
    response.raise_for_status()
    openid_config = response.json()
    jwks_uri = openid_config["jwks_uri"]
    
    return PyJWKClient(jwks_uri)


def verify_token(token: str) -> dict:
    """
    Verify and decode a Microsoft Entra JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        config = get_auth_config()
        jwks_client = get_jwks_client()
        
        # Get the signing key from the JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Decode and verify the token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=config.client_id,
            issuer=config.issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True,
            }
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """
    Dependency to get the current authenticated user
    
    Args:
        credentials: HTTP Authorization credentials from request
        
    Returns:
        User information from token payload
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    # Extract user information from token
    user_info = {
        "user_id": payload.get("oid"),  # Object ID (unique user identifier)
        "email": payload.get("preferred_username") or payload.get("email"),
        "name": payload.get("name"),
        "tenant_id": payload.get("tid"),
        "app_id": payload.get("azp") or payload.get("appid"),
        "roles": payload.get("roles", []),
        "scopes": payload.get("scp", "").split() if payload.get("scp") else [],
    }
    
    return user_info


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[dict]:
    """
    Dependency to get the current user if authenticated, None otherwise
    Useful for endpoints that have optional authentication
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency factory to require a specific role
    
    Usage:
        @app.get("/admin", dependencies=[Depends(require_role("Admin"))])
    """
    async def role_checker(user: dict = Depends(get_current_user)):
        roles = user.get("roles", [])
        if required_role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"User does not have required role: {required_role}"
            )
        return user
    
    return role_checker


def require_scope(required_scope: str):
    """
    Dependency factory to require a specific scope
    
    Usage:
        @app.get("/api/data", dependencies=[Depends(require_scope("access_as_user"))])
    """
    async def scope_checker(user: dict = Depends(get_current_user)):
        scopes = user.get("scopes", [])
        if required_scope not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"User does not have required scope: {required_scope}"
            )
        return user
    
    return scope_checker
