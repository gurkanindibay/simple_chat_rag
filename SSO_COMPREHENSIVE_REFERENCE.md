# SSO Comprehensive Reference Guide

> **Complete Technical Documentation for Microsoft Entra ID Single Sign-On Implementation**  
> Version: 2.0  
> Last Updated: October 20, 2025  
> Status: ✅ Production Ready

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Authentication Flow](#authentication-flow)
4. [Implementation Details](#implementation-details)
5. [Security Model](#security-model)
6. [Cross-Application SSO](#cross-application-sso)
7. [Token Management](#token-management)
8. [Logout Architecture](#logout-architecture)
9. [Configuration Reference](#configuration-reference)
10. [API Integration](#api-integration)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Best Practices](#best-practices)

---

## Executive Summary

### What is SSO?

**Single Sign-On (SSO)** allows users to authenticate once and gain access to multiple applications without re-entering credentials. This implementation uses **Microsoft Entra ID (formerly Azure Active Directory)** as the identity provider.

### Key Benefits

✅ **User Experience**
- Sign in once, access all applications
- No password fatigue
- Seamless cross-application navigation

✅ **Security**
- Centralized identity management
- Multi-factor authentication (MFA) support
- Role-based access control (RBAC)
- Automated token validation

✅ **Administration**
- Single point of user management
- Automated provisioning/deprovisioning
- Comprehensive audit trails
- Compliance-ready

### Implementation Scope

This implementation includes:

1. **Frontend (React SPA)** - MSAL.js integration for browser-based authentication
2. **Backend (FastAPI)** - JWT token validation and role-based authorization
3. **SSO Showcase SPA** - Standalone demonstration of cross-application SSO
4. **Cross-App Logout** - Coordinated logout across multiple applications

---

## Architecture Overview

### High-Level Architecture

```mermaid
graph TB
    subgraph "User's Browser"
        A[User]
        B[Main App<br/>React SPA<br/>Port 5173]
        C[SSO Showcase<br/>Vanilla JS SPA<br/>Port 8001]
        D[localStorage<br/>Token Cache]
    end
    
    subgraph "Microsoft Cloud"
        E[Microsoft Entra ID<br/>Identity Provider]
        F[Token Endpoint]
        G[JWKS Endpoint<br/>Public Keys]
    end
    
    subgraph "Backend Services"
        H[FastAPI Server<br/>Port 8000]
        I[JWT Validator]
        J[PostgreSQL<br/>Application Data]
    end
    
    A -->|1. Initiate Login| B
    B -->|2. Redirect to Login| E
    E -->|3. Authenticate| A
    A -->|4. Credentials| E
    E -->|5. Authorization Code| B
    B -->|6. Exchange Code| F
    F -->|7. ID Token + Access Token| B
    B -->|8. Store Tokens| D
    B -->|9. API Request + Token| H
    H -->|10. Verify Signature| G
    H -->|11. Validate Token| I
    I -->|12. Query Data| J
    J -->|13. Response| H
    H -->|14. Protected Data| B
    
    C -->|SSO: Read Tokens| D
    C -->|API Request + Token| H
    
    style E fill:#0078d4,color:#fff
    style D fill:#ffc107,color:#000
    style I fill:#28a745,color:#fff
```

### Component Architecture

```mermaid
graph LR
    subgraph "Frontend Layer"
        A[React Components]
        B[MSAL Provider]
        C[Auth Context]
        D[API Client]
    end
    
    subgraph "Authentication"
        E[MSAL Browser]
        F[Token Cache]
        G[Account Manager]
    end
    
    subgraph "Backend Layer"
        H[FastAPI Routes]
        I[Auth Middleware]
        J[JWT Verifier]
    end
    
    subgraph "Identity Provider"
        K[Entra ID]
        L[OAuth 2.0]
        M[OpenID Connect]
    end
    
    A --> B
    B --> E
    E --> F
    E --> G
    A --> D
    D --> H
    H --> I
    I --> J
    J --> K
    E <--> K
    K --> L
    K --> M
    
    style B fill:#61dafb,color:#000
    style E fill:#0078d4,color:#fff
    style I fill:#009485,color:#fff
    style K fill:#0078d4,color:#fff
```

---

## Authentication Flow

### OAuth 2.0 Authorization Code Flow with PKCE

This implementation uses the **Authorization Code Flow with Proof Key for Code Exchange (PKCE)**, which is the recommended flow for Single Page Applications (SPAs).

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant SPA as React SPA
    participant MSAL as MSAL.js Library
    participant Browser as Browser Storage
    participant Entra as Microsoft Entra ID
    participant API as FastAPI Backend
    
    User->>SPA: Click "Sign In"
    SPA->>MSAL: loginPopup() or loginRedirect()
    
    Note over MSAL: Generate PKCE challenge
    MSAL->>MSAL: code_verifier = random()
    MSAL->>MSAL: code_challenge = SHA256(code_verifier)
    
    MSAL->>Browser: Store code_verifier in sessionStorage
    MSAL->>Entra: Authorization Request<br/>(client_id, redirect_uri, code_challenge)
    
    Entra->>User: Display Login Page
    User->>Entra: Enter Credentials + MFA
    Entra->>Entra: Validate User
    
    Entra->>MSAL: Authorization Code
    MSAL->>Browser: Retrieve code_verifier
    MSAL->>Entra: Token Request<br/>(code, code_verifier)
    
    Entra->>Entra: Validate code_verifier matches code_challenge
    Entra->>MSAL: ID Token + Access Token + Refresh Token
    
    MSAL->>Browser: Store tokens in localStorage (encrypted)
    MSAL->>SPA: Authentication Success + Account Object
    
    SPA->>API: Request + Access Token in Authorization Header
    API->>Entra: Fetch Public Keys (JWKS)
    API->>API: Verify Token Signature
    API->>API: Validate Claims (aud, iss, exp)
    API->>SPA: Protected Resource
    
    Note over SPA,Browser: Token expires after 1 hour
    
    SPA->>MSAL: acquireTokenSilent() - need new token
    MSAL->>Entra: Refresh Token Request
    Entra->>MSAL: New Access Token
    MSAL->>Browser: Update cache
    MSAL->>SPA: New token ready
```

### Login Flow Variations

#### Popup Login Flow

```mermaid
sequenceDiagram
    participant User
    participant MainWindow as Main Window
    participant Popup as Login Popup
    participant Entra as Microsoft Entra ID
    
    User->>MainWindow: Click "Sign in with Microsoft (Popup)"
    MainWindow->>Popup: Open popup window
    Popup->>Entra: Redirect to login.microsoftonline.com
    Entra->>User: Show login form in popup
    User->>Entra: Enter credentials
    Entra->>Popup: Return tokens to popup
    Popup->>MainWindow: Post message with auth result
    MainWindow->>MainWindow: Store tokens in localStorage
    Popup->>Popup: Close popup window
    MainWindow->>User: Show authenticated UI
    
    Note over MainWindow: Main window never navigates away
```

**Advantages:**
- Main window state preserved
- Better UX (no page reload)
- Faster perceived performance

**Disadvantages:**
- Popup blockers may interfere
- More complex error handling

#### Redirect Login Flow

```mermaid
sequenceDiagram
    participant User
    participant SPA as React SPA
    participant Entra as Microsoft Entra ID
    
    User->>SPA: Click "Sign in with Microsoft (Redirect)"
    SPA->>SPA: Save application state
    SPA->>Entra: Full page redirect
    Entra->>User: Show login form
    User->>Entra: Enter credentials
    Entra->>SPA: Redirect back with auth code
    SPA->>SPA: handleRedirectPromise()
    SPA->>SPA: Exchange code for tokens
    SPA->>SPA: Restore application state
    SPA->>User: Show authenticated UI
    
    Note over SPA: Page reloads during authentication
```

**Advantages:**
- No popup blocker issues
- Simpler to implement
- Better mobile support

**Disadvantages:**
- Page state is lost
- Slower user experience

### Silent Token Acquisition

```mermaid
sequenceDiagram
    participant SPA
    participant MSAL as MSAL.js
    participant Cache as Token Cache
    participant iframe as Hidden iframe
    participant Entra as Microsoft Entra ID
    
    SPA->>MSAL: acquireTokenSilent({ scopes })
    MSAL->>Cache: Check for valid cached token
    
    alt Token found and valid
        Cache->>MSAL: Return cached token
        MSAL->>SPA: Token (from cache)
    else Token expired or missing
        MSAL->>iframe: Create hidden iframe
        iframe->>Entra: Silent auth request (using cookies)
        
        alt User has active session
            Entra->>iframe: Return new token
            iframe->>MSAL: New token
            MSAL->>Cache: Update cache
            MSAL->>SPA: New token
        else No active session
            Entra->>MSAL: Error (interaction required)
            MSAL->>SPA: InteractionRequiredAuthError
            SPA->>User: Redirect to login
        end
    end
```

---

## Implementation Details

### Frontend Implementation

#### 1. MSAL Configuration (`authConfig.js`)

```javascript
import { LogLevel } from '@azure/msal-browser'

export const msalConfig = {
  auth: {
    clientId: process.env.VITE_AZURE_FRONTEND_CLIENT_ID,
    authority: `https://login.microsoftonline.com/${process.env.VITE_AZURE_FRONTEND_TENANT_ID}`,
    redirectUri: window.location.origin,
  },
  cache: {
    cacheLocation: 'localStorage',  // Enable SSO across tabs
    storeAuthStateInCookie: true,   // Safari compatibility
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return;
        console.log(`[MSAL][${level}]`, message);
      },
      piiLoggingEnabled: false,
    },
  },
}

export const loginRequest = {
  scopes: ['openid', 'profile', 'User.Read'],
}

export const tokenRequest = {
  scopes: process.env.VITE_AZURE_BACKEND_SCOPE 
    ? [process.env.VITE_AZURE_BACKEND_SCOPE]
    : ['User.Read'],
}
```

#### 2. MSAL Initialization (`main.jsx`)

```javascript
import { PublicClientApplication } from '@azure/msal-browser'
import { MsalProvider } from '@azure/msal-react'

const msalInstance = new PublicClientApplication(msalConfig);

const initAndRender = async () => {
  // Initialize MSAL
  await msalInstance.initialize();
  
  // Handle redirect response
  const redirectResult = await msalInstance.handleRedirectPromise();
  
  if (redirectResult && redirectResult.account) {
    msalInstance.setActiveAccount(redirectResult.account);
  }
  
  // Check for cached accounts
  const accounts = msalInstance.getAllAccounts();
  if (accounts.length > 0) {
    msalInstance.setActiveAccount(accounts[0]);
  }
  
  // Render app
  createRoot(document.getElementById('root')).render(
    <MsalProvider instance={msalInstance}>
      <App />
    </MsalProvider>
  );
};

initAndRender();
```

#### 3. Login Component

```javascript
import { useMsal } from '@azure/msal-react';
import { loginRequest } from '../authConfig';

export const Login = () => {
  const { instance } = useMsal();

  const handleLogin = (loginType) => {
    // Clear sessionStorage to prevent interaction_in_progress errors
    const sessKeys = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      sessKeys.push(sessionStorage.key(i));
    }
    sessKeys.forEach(k => {
      if (k && (k.includes('msal') || k.includes('login.windows'))) {
        sessionStorage.removeItem(k);
      }
    });

    if (loginType === 'popup') {
      instance.loginPopup(loginRequest)
        .then((response) => {
          instance.setActiveAccount(response.account);
        })
        .catch((error) => console.error('Login error:', error));
    } else {
      instance.loginRedirect(loginRequest)
        .catch((error) => console.error('Login error:', error));
    }
  };

  return (
    <div className="login-container">
      <button onClick={() => handleLogin('popup')}>
        Sign in with Microsoft (Popup)
      </button>
      <button onClick={() => handleLogin('redirect')}>
        Sign in with Microsoft (Redirect)
      </button>
    </div>
  );
};
```

#### 4. Protected API Calls

```javascript
import { msalInstance } from './msalInstance';
import { tokenRequest } from '../authConfig';

export async function callProtectedAPI(endpoint, options = {}) {
  try {
    // Acquire token silently
    const response = await msalInstance.acquireTokenSilent({
      ...tokenRequest,
      account: msalInstance.getActiveAccount(),
    });

    // Make API call with token
    const result = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${response.accessToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (!result.ok) {
      throw new Error(`API error: ${result.status}`);
    }

    return await result.json();
  } catch (error) {
    if (error instanceof InteractionRequiredAuthError) {
      // Token expired, need user interaction
      await msalInstance.acquireTokenPopup(tokenRequest);
      // Retry the call
      return callProtectedAPI(endpoint, options);
    }
    throw error;
  }
}
```

### Backend Implementation

#### 1. JWT Token Validation (`auth.py`)

```python
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

class EntraAuthConfig:
    def __init__(self):
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"
    
    @property
    def issuer(self):
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"

def verify_token(token: str) -> dict:
    """Verify and decode a Microsoft Entra JWT token"""
    config = get_auth_config()
    jwks_client = get_jwks_client()
    
    # Get signing key
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    
    # Verify token
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
```

#### 2. Protected Endpoints

```python
from fastapi import Depends
from backend.auth import get_current_user, require_role

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(require_role('rag_chat_user'))):
    """Protected endpoint requiring rag_chat_user role"""
    return {"user": current_user}

@app.post("/chat")
async def chat(
    req: ChatRequest, 
    current_user: dict = Depends(require_role('rag_chat_user'))
):
    """Chat endpoint - requires authentication and role"""
    # User is authenticated and has required role
    return process_chat(req.question, current_user)
```

#### 3. Token Claims Extraction

```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Extract user information from validated token"""
    token = credentials.credentials
    payload = verify_token(token)
    
    return {
        "user_id": payload.get("oid"),      # Object ID
        "email": payload.get("preferred_username"),
        "name": payload.get("name"),
        "tenant_id": payload.get("tid"),
        "roles": payload.get("roles", []),
        "scopes": payload.get("scp", "").split(),
    }
```

---

## Security Model

### Defense in Depth

```mermaid
graph TD
    A[User Request] --> B{HTTPS?}
    B -->|No| C[Reject]
    B -->|Yes| D{Token Present?}
    D -->|No| C
    D -->|Yes| E{Token Valid Signature?}
    E -->|No| C
    E -->|Yes| F{Token Not Expired?}
    F -->|No| C
    F -->|Yes| G{Correct Audience?}
    G -->|No| C
    G -->|Yes| H{Correct Issuer?}
    H -->|No| C
    H -->|Yes| I{User Has Role?}
    I -->|No| C
    I -->|Yes| J{User Has Scope?}
    J -->|No| C
    J -->|Yes| K[Process Request]
    
    style C fill:#dc3545,color:#fff
    style K fill:#28a745,color:#fff
```

### Token Validation Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Cache as JWKS Cache
    participant Entra as Microsoft Entra ID
    
    Client->>API: Request + Bearer Token
    API->>API: Extract token from Authorization header
    
    API->>Cache: Get cached public keys
    
    alt Keys cached and valid
        Cache->>API: Return public keys
    else Keys expired or missing
        API->>Entra: Fetch JWKS (public keys)
        Entra->>API: Return public keys
        API->>Cache: Update cache
    end
    
    API->>API: Decode JWT header (kid)
    API->>API: Find matching public key
    API->>API: Verify signature using RS256
    
    alt Signature invalid
        API->>Client: 401 Unauthorized
    end
    
    API->>API: Validate audience (aud claim)
    API->>API: Validate issuer (iss claim)
    API->>API: Validate expiration (exp claim)
    API->>API: Validate not before (nbf claim)
    
    alt Any validation fails
        API->>Client: 401 Unauthorized
    end
    
    API->>API: Extract user claims
    API->>API: Check roles array
    API->>API: Check scopes
    
    alt Missing required role/scope
        API->>Client: 403 Forbidden
    end
    
    API->>Client: 200 OK + Protected Resource
```

### JWT Token Structure

```mermaid
graph LR
    A[JWT Token] --> B[Header]
    A --> C[Payload]
    A --> D[Signature]
    
    B --> B1[alg: RS256]
    B --> B2[typ: JWT]
    B --> B3[kid: key-id]
    
    C --> C1[iss: issuer]
    C --> C2[aud: audience]
    C --> C3[exp: expiration]
    C --> C4[iat: issued at]
    C --> C5[sub: subject]
    C --> C6[oid: user ID]
    C --> C7[roles: array]
    C --> C8[scp: scopes]
    
    D --> D1[RSASHA256<br/>signature]
    
    style A fill:#0078d4,color:#fff
    style B fill:#ffc107,color:#000
    style C fill:#28a745,color:#fff
    style D fill:#dc3545,color:#fff
```

### Security Best Practices Implemented

| Layer | Security Control | Status |
|-------|-----------------|--------|
| **Transport** | HTTPS/TLS 1.2+ | ✅ Required in production |
| **Authentication** | OAuth 2.0 + OIDC | ✅ Implemented |
| **Authorization** | Role-Based Access Control | ✅ Implemented |
| **Token Security** | JWT Signature Verification | ✅ Implemented |
| **Token Expiration** | 1-hour access tokens | ✅ Enforced |
| **Refresh Tokens** | Secure rotation | ✅ MSAL handles |
| **PKCE** | Code injection protection | ✅ Implemented |
| **CORS** | Restricted origins | ✅ Configured |
| **Session Storage** | Encrypted cache | ✅ MSAL v3+ |
| **XSS Protection** | HttpOnly cookies | ✅ Configured |
| **CSRF Protection** | State parameter | ✅ MSAL handles |

---

## Cross-Application SSO

### Two-SPA Architecture

```mermaid
graph TB
    subgraph "Browser Environment"
        LS[localStorage<br/>Shared Token Storage]
        
        subgraph "Main Application"
            A1[React SPA]
            A2[MSAL Instance 1]
            A3[Client ID:<br/>a8a16485-0827-46c6]
        end
        
        subgraph "SSO Showcase"
            B1[Vanilla JS SPA]
            B2[MSAL Instance 2]
            B3[Client ID:<br/>630f781d-5e19-46c4]
        end
    end
    
    subgraph "Microsoft Entra ID"
        C[Shared User Session]
    end
    
    A1 --> A2
    A2 --> A3
    A2 --> LS
    A2 --> C
    
    B1 --> B2
    B2 --> B3
    B2 --> LS
    B2 --> C
    
    LS -.->|SSO Magic| C
    
    style LS fill:#ffc107,color:#000
    style C fill:#0078d4,color:#fff
    style A3 fill:#61dafb,color:#000
    style B3 fill:#28a745,color:#fff
```

### SSO Enablement Flow

```mermaid
sequenceDiagram
    participant User
    participant MainApp as Main App<br/>(Port 5173)
    participant Storage as localStorage
    participant Showcase as SSO Showcase<br/>(Port 8001)
    participant Entra as Microsoft Entra ID
    
    Note over User,Entra: Step 1: User signs in to Main App
    
    User->>MainApp: Navigate to app
    MainApp->>User: Show login button
    User->>MainApp: Click "Sign in"
    MainApp->>Entra: Redirect to login
    User->>Entra: Enter credentials
    Entra->>MainApp: Return tokens
    MainApp->>Storage: Store tokens (MSAL cache)
    MainApp->>User: Show authenticated UI
    
    Note over User,Entra: Step 2: User opens SSO Showcase
    
    User->>Showcase: Navigate to http://localhost:8001
    Showcase->>Showcase: Page loads
    Showcase->>Showcase: Call ssoSilent()
    Showcase->>Storage: Read MSAL tokens
    
    alt Tokens found in localStorage
        Storage->>Showcase: Return cached session
        Showcase->>Entra: Validate session (hidden iframe)
        Entra->>Showcase: Session valid, return tokens
        Showcase->>User: ✅ Automatically authenticated!
    else No tokens found
        Showcase->>User: Show login button
        User->>Showcase: Manual login required
    end
    
    Note over User,Showcase: SSO achieved - no password entry needed!
```

### Critical SSO Configuration

**Why localStorage is Essential:**

```javascript
// ✅ CORRECT - Enables SSO
const msalConfig = {
  cache: {
    cacheLocation: 'localStorage',  // Shared across tabs/windows
  }
}

// ❌ WRONG - Breaks SSO
const msalConfig = {
  cache: {
    cacheLocation: 'sessionStorage',  // Isolated per tab
  }
}
```

**How ssoSilent() Works:**

```javascript
// In SSO Showcase App
async function attemptSSOSilent() {
  try {
    // This reads from localStorage and validates with Entra ID
    const response = await msalInstance.ssoSilent({
      scopes: ['openid', 'profile', 'User.Read']
    });
    
    // Success! User is authenticated without password
    console.log('SSO successful:', response.account);
    showAuthenticatedUI(response.account);
    
  } catch (error) {
    // No existing session - show login button
    console.log('SSO failed, manual login required');
    showLoginButton();
  }
}
```

### SSO Troubleshooting Decision Tree

```mermaid
graph TD
    A[SSO Not Working] --> B{Same Browser?}
    B -->|No| C[SSO only works<br/>within same browser]
    B -->|Yes| D{localStorage enabled?}
    D -->|No| E[Enable localStorage<br/>in browser settings]
    D -->|Yes| F{cacheLocation set to<br/>localStorage?}
    F -->|No| G[Change config to<br/>localStorage]
    F -->|Yes| H{Same tenant?}
    H -->|No| I[Apps must be in<br/>same Azure AD tenant]
    H -->|Yes| J{Using ssoSilent?}
    J -->|No| K[Use ssoSilent()<br/>not loginRedirect()]
    J -->|Yes| L{Check browser console}
    L --> M[Review MSAL logs<br/>for specific error]
    
    style A fill:#dc3545,color:#fff
    style C fill:#ffc107,color:#000
    style E fill:#ffc107,color:#000
    style G fill:#28a745,color:#fff
    style I fill:#ffc107,color:#000
    style K fill:#28a745,color:#fff
    style M fill:#0078d4,color:#fff
```

---

## Token Management

### Token Lifecycle

```mermaid
stateDiagram-v2
    [*] --> NoToken: User not authenticated
    NoToken --> Acquiring: User clicks login
    Acquiring --> Active: Token received
    Active --> Active: Token valid (< 60 min)
    Active --> Refreshing: Token expired
    Refreshing --> Active: Refresh successful
    Refreshing --> NoToken: Refresh failed
    Active --> NoToken: User logs out
    NoToken --> [*]
    
    note right of Active
        Access Token TTL: 1 hour
        Refresh Token TTL: 24 hours (default)
        Silent refresh: acquireTokenSilent()
    end note
```

### Token Acquisition Strategies

```mermaid
graph TD
    A[Need Access Token] --> B{Token in cache?}
    B -->|Yes| C{Token valid?}
    B -->|No| D[Acquire new token]
    
    C -->|Yes| E[Use cached token]
    C -->|No| F[Token expired]
    
    F --> G{Refresh token valid?}
    G -->|Yes| H[acquireTokenSilent]
    G -->|No| I[Interaction required]
    
    H --> J{Silent refresh success?}
    J -->|Yes| E
    J -->|No| I
    
    I --> K[acquireTokenPopup or<br/>acquireTokenRedirect]
    K --> E
    
    D --> K
    
    style E fill:#28a745,color:#fff
    style I fill:#ffc107,color:#000
    style K fill:#0078d4,color:#fff
```

### Access Token vs ID Token vs Refresh Token

| Token Type | Purpose | Lifetime | Storage | Usage |
|------------|---------|----------|---------|-------|
| **ID Token** | User identity | 1 hour | localStorage | Display user info |
| **Access Token** | API authorization | 1 hour | localStorage | Bearer token in API calls |
| **Refresh Token** | Get new access tokens | 24 hours+ | localStorage (encrypted) | Silent token renewal |

### Token Refresh Flow

```mermaid
sequenceDiagram
    participant App as React App
    participant MSAL as MSAL.js
    participant Cache as Token Cache
    participant Entra as Microsoft Entra ID
    participant API as Backend API
    
    Note over App,API: Scenario: Access token expired
    
    App->>API: Request with expired token
    API->>App: 401 Unauthorized
    
    App->>MSAL: acquireTokenSilent()
    MSAL->>Cache: Check access token
    Cache->>MSAL: Token expired
    
    MSAL->>Cache: Check refresh token
    Cache->>MSAL: Refresh token valid
    
    MSAL->>Entra: Token refresh request<br/>(refresh_token)
    Entra->>Entra: Validate refresh token
    Entra->>MSAL: New access token + refresh token
    
    MSAL->>Cache: Update tokens
    MSAL->>App: New access token
    
    App->>API: Retry request with new token
    API->>App: 200 OK + Data
    
    Note over App,MSAL: All happens silently,<br/>user sees no interruption
```

---

## Logout Architecture

### Global Logout Flow

```mermaid
sequenceDiagram
    participant User
    participant App1 as Main App
    participant App2 as SSO Showcase
    participant Storage as localStorage
    participant Session as sessionStorage
    participant Entra as Microsoft Entra ID
    
    User->>App1: Click "Logout"
    
    Note over App1,Storage: Step 1: Set global logout flags
    
    App1->>Storage: Set msal_global_logout timestamp
    App1->>Storage: Set msal_global_logout_processed
    App1->>Storage: Remove app_logged_in
    
    Note over App1,Session: Step 2: Clear all MSAL storage
    
    App1->>Storage: Remove all msal.* keys
    App1->>Storage: Remove login.windows.* keys
    App1->>Session: Remove all msal.* keys
    App1->>Session: Remove login.windows.* keys
    
    Note over App1,Entra: Step 3: Perform MSAL logout
    
    App1->>Entra: logoutPopup() or logoutRedirect()
    Entra->>Entra: Clear server-side session
    Entra->>App1: Logout complete
    App1->>User: Show login screen
    
    Note over App2,Storage: Step 4: Other apps detect logout
    
    loop Every 5 seconds
        App2->>Storage: Check msal_global_logout flag
        Storage->>App2: Logout flag found
        App2->>App2: Logout timestamp > last processed?
        App2->>Storage: Clear MSAL tokens
        App2->>Entra: logoutRedirect()
        App2->>User: Show login screen
    end
    
    Note over User,Entra: All apps now logged out
```

### Logout Coordination Mechanism

```mermaid
graph TD
    A[User logs out from App A] --> B[setGlobalLogoutFlags]
    B --> C[Set msal_global_logout = timestamp]
    B --> D[Set logout cookie]
    
    E[App B running in another tab] --> F[Periodic check every 5s]
    F --> G{msal_global_logout flag exists?}
    G -->|No| F
    G -->|Yes| H{Timestamp > last_processed?}
    H -->|No| I[Ignore - already processed]
    H -->|Yes| J[Trigger logout in App B]
    
    J --> K[Clear MSAL storage]
    J --> L[Call logoutRedirect]
    J --> M[Mark as processed]
    
    M --> N[Update msal_global_logout_processed]
    M --> O[Remove msal_global_logout flag]
    
    style A fill:#dc3545,color:#fff
    style J fill:#ffc107,color:#000
    style N fill:#28a745,color:#fff
```

### useLogout Hook Implementation

```javascript
import { useState } from 'react';
import { useMsal } from '@azure/msal-react';
import { setGlobalLogoutFlags, clearMSALStorage } from '../utils/logout';

export function useLogout(options = {}) {
  const { logoutType = 'popup', postLogoutRedirectUri = '/' } = options;
  const { instance } = useMsal();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const logout = async () => {
    if (isLoggingOut) return;
    setIsLoggingOut(true);
    
    try {
      // Set flags for other apps
      setGlobalLogoutFlags();
      
      // Clear all MSAL storage
      clearMSALStorage();
      
      // Perform MSAL logout
      if (logoutType === 'redirect') {
        await instance.logoutRedirect({ postLogoutRedirectUri });
      } else {
        await instance.logoutPopup({ postLogoutRedirectUri });
      }
    } catch (error) {
      console.error('Logout error:', error);
      throw error;
    } finally {
      setIsLoggingOut(false);
    }
  };

  return { logout, isLoggingOut };
}
```

### Logout Storage Cleanup

```javascript
export function clearMSALStorage() {
  // Clear localStorage (except logout coordination flags)
  const keysToRemove = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && (key.includes('msal') || key.includes('login.windows')) && 
        key !== 'msal_global_logout' && 
        key !== 'msal_global_logout_processed') {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach(key => localStorage.removeItem(key));
  
  // Clear ALL sessionStorage (no exceptions)
  const sessionKeys = [];
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (key && (key.includes('msal') || key.includes('login.windows'))) {
      sessionKeys.push(key);
    }
  }
  sessionKeys.forEach(key => sessionStorage.removeItem(key));
  
  // Clear cookies
  document.cookie.split(';').forEach(cookie => {
    const cookieName = cookie.split('=')[0].trim();
    if ((cookieName.includes('msal') || cookieName.includes('login.windows')) && 
        cookieName !== 'msal_global_logout') {
      document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
    }
  });
}
```

---

## Configuration Reference

### Environment Variables

#### Frontend Configuration (`.env`)

```bash
# Azure AD Frontend SPA Configuration
VITE_AZURE_FRONTEND_CLIENT_ID=a8a16485-0827-46c6-b3e0-91fca5966341
VITE_AZURE_FRONTEND_TENANT_ID=066690f2-a8a6-4889-852e-124371dcbd6f

# Redirect URI (automatically set to window.location.origin)
VITE_REDIRECT_URI=http://localhost:5173

# Backend API Scope (optional - for calling custom backend API)
VITE_AZURE_BACKEND_CLIENT_ID=backend-client-id-here
VITE_AZURE_BACKEND_SCOPE=api://backend-client-id/access_as_user
```

#### Backend Configuration (`.env`)

```bash
# Azure AD Backend API Configuration
AZURE_TENANT_ID=066690f2-a8a6-4889-852e-124371dcbd6f
AZURE_CLIENT_ID=backend-client-id-here
AZURE_CLIENT_SECRET=optional-for-backend-flows

# Database Configuration
POSTGRES_URL=postgresql://user:pass@localhost:5432/dbname
```

### Azure AD App Registration Settings

#### Frontend SPA Registration

```yaml
Name: "RAG Chat Frontend"
Application Type: Single-page application (SPA)
Client ID: a8a16485-0827-46c6-b3e0-91fca5966341
Supported Account Types: Single tenant
Redirect URIs:
  - http://localhost:5173
  - http://localhost:5173/
  - https://yourdomain.com
  - https://yourdomain.com/

Authentication:
  Platform: Single-page application
  Allow public client flows: No
  
API Permissions:
  - Microsoft Graph > User.Read (Delegated)
  - [Your Backend API] > access_as_user (Delegated)
  
Token Configuration:
  Optional Claims:
    - email
    - preferred_username
  App Roles:
    - rag_chat_user
```

#### Backend API Registration

```yaml
Name: "RAG Chat Backend API"
Application Type: Web
Client ID: backend-client-id-here
Supported Account Types: Single tenant

Expose an API:
  Application ID URI: api://backend-client-id-here
  Scopes:
    - access_as_user (Admins and users)
    
App Roles:
  Name: rag_chat_user
  Allowed Member Types: Users/Groups
  Value: rag_chat_user
  Description: Access to RAG chat features
```

### MSAL Configuration Options

```javascript
const msalConfig = {
  auth: {
    clientId: string,              // Required
    authority: string,             // Required
    redirectUri: string,           // Required
    postLogoutRedirectUri: string, // Optional
    navigateToLoginRequestUrl: boolean, // Default: true
    clientCapabilities: string[],  // Optional: ["CP1"]
  },
  cache: {
    cacheLocation: 'localStorage' | 'sessionStorage', // Default: 'sessionStorage'
    storeAuthStateInCookie: boolean, // Default: false
    secureCookies: boolean,        // Default: false
    cacheMigrationEnabled: boolean, // Default: false
    claimsBasedCachingEnabled: boolean, // Default: false
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => void,
      logLevel: LogLevel,
      piiLoggingEnabled: boolean,
    },
    windowHashTimeout: number,     // Default: 6000
    iframeHashTimeout: number,     // Default: 6000
    loadFrameTimeout: number,      // Default: 6000
    asyncPopups: boolean,          // Default: false
  },
  telemetry: {
    application: {
      appName: string,
      appVersion: string,
    },
  },
}
```

---

## API Integration

### API Request Flow with Authentication

```mermaid
sequenceDiagram
    participant Component as React Component
    participant Hook as useChatAPI Hook
    participant Client as API Client
    participant MSAL as MSAL.js
    participant API as FastAPI Backend
    
    Component->>Hook: sendMessage("Hello")
    Hook->>Client: callAPI('/chat', { question: "Hello" })
    
    Client->>MSAL: acquireTokenSilent()
    
    alt Token available
        MSAL->>Client: Return access token
    else Need refresh
        MSAL->>MSAL: Refresh token silently
        MSAL->>Client: Return new access token
    else Refresh failed
        MSAL->>Component: Show login modal
        Component->>MSAL: User re-authenticates
        MSAL->>Client: Return access token
    end
    
    Client->>API: POST /chat<br/>Authorization: Bearer {token}
    API->>API: Verify token signature
    API->>API: Check roles & scopes
    API->>API: Process request
    API->>Client: 200 OK + Response
    Client->>Hook: Return response
    Hook->>Component: Update UI
```

### API Client Implementation

```javascript
// api/client.js
import { InteractionRequiredAuthError } from '@azure/msal-browser';
import { tokenRequest } from '../authConfig';

let msalInstance = null;

export function setMsalInstance(instance) {
  msalInstance = instance;
}

async function getAccessToken() {
  if (!msalInstance) {
    throw new Error('MSAL instance not initialized');
  }

  const account = msalInstance.getActiveAccount();
  if (!account) {
    throw new Error('No active account');
  }

  try {
    const response = await msalInstance.acquireTokenSilent({
      ...tokenRequest,
      account,
    });
    return response.accessToken;
  } catch (error) {
    if (error instanceof InteractionRequiredAuthError) {
      // Try popup if silent fails
      const response = await msalInstance.acquireTokenPopup(tokenRequest);
      return response.accessToken;
    }
    throw error;
  }
}

export async function apiRequest(endpoint, options = {}) {
  const token = await getAccessToken();
  
  const response = await fetch(`${import.meta.env.VITE_API_URL}${endpoint}`, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  if (response.status === 401) {
    // Token might be invalid, try to get a new one
    const newToken = await getAccessToken();
    const retryResponse = await fetch(`${import.meta.env.VITE_API_URL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${newToken}`,
        'Content-Type': 'application/json',
      },
    });
    return retryResponse;
  }

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}
```

### Protected Endpoint Examples

```python
# Different authorization levels

# 1. Public endpoint (no auth)
@app.get("/public/info")
async def public_info():
    return {"message": "This is public"}

# 2. Authenticated endpoint (any valid token)
@app.get("/user/profile")
async def user_profile(user: dict = Depends(get_current_user)):
    return {"user": user}

# 3. Role-required endpoint
@app.get("/admin/users")
async def admin_users(user: dict = Depends(require_role('admin'))):
    return {"users": get_all_users()}

# 4. Scope-required endpoint
@app.get("/api/data")
async def api_data(user: dict = Depends(require_scope('access_as_user'))):
    return {"data": get_data()}

# 5. Multiple requirements
@app.post("/admin/delete")
async def admin_delete(
    user: dict = Depends(require_role('admin')),
    verified: dict = Depends(require_scope('write_access'))
):
    return {"status": "deleted"}
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: "interaction_in_progress" Error

**Symptoms:**
```
BrowserAuthError: interaction_in_progress: Interaction is currently in progress.
```

**Cause:** Stale state in sessionStorage from previous incomplete auth flow.

**Solution:**
```javascript
// Clear sessionStorage before login
const sessionKeys = [];
for (let i = 0; i < sessionStorage.length; i++) {
  sessionKeys.push(sessionStorage.key(i));
}
sessionKeys.forEach(k => {
  if (k && (k.includes('msal') || k.includes('login.windows'))) {
    sessionStorage.removeItem(k);
  }
});

// Then proceed with login
instance.loginPopup(loginRequest);
```

#### Issue 2: Tokens Not Persisting After Page Reload

**Diagnosis Flow:**

```mermaid
graph TD
    A[Tokens disappear on reload] --> B{cacheLocation setting?}
    B -->|sessionStorage| C[Change to localStorage]
    B -->|localStorage| D{MSAL initialized?}
    D -->|No| E[await instance.initialize]
    D -->|Yes| F{Calling getAllAccounts?}
    F -->|No| G[Call getAllAccounts<br/>after init]
    F -->|Yes| H{Cache encrypted?}
    H -->|Yes| I[Wait for decryption<br/>100-500ms delay]
    
    style C fill:#28a745,color:#fff
    style E fill:#28a745,color:#fff
    style G fill:#28a745,color:#fff
    style I fill:#ffc107,color:#000
```

**Solution:**
```javascript
// main.jsx
const msalInstance = new PublicClientApplication(msalConfig);

const initAndRender = async () => {
  // 1. Initialize first
  await msalInstance.initialize();
  
  // 2. Handle redirect
  await msalInstance.handleRedirectPromise();
  
  // 3. Wait for cache decryption (MSAL v3+)
  await new Promise(resolve => setTimeout(resolve, 100));
  
  // 4. Now get accounts
  const accounts = msalInstance.getAllAccounts();
  if (accounts.length > 0) {
    msalInstance.setActiveAccount(accounts[0]);
  }
  
  // 5. Render
  render(<App />);
};

initAndRender();
```

#### Issue 3: 401 Unauthorized from Backend

**Troubleshooting Steps:**

```mermaid
graph TD
    A[401 Error] --> B{Token present in request?}
    B -->|No| C[Check API client<br/>adding Authorization header]
    B -->|Yes| D{Token format correct?}
    D -->|No| E[Should be:<br/>Authorization: Bearer token]
    D -->|Yes| F{Token expired?}
    F -->|Yes| G[Implement token refresh]
    F -->|No| H{Correct audience?}
    H -->|No| I[Check token aud claim<br/>matches backend clientId]
    H -->|Yes| J{Signature valid?}
    J -->|No| K[Check JWKS endpoint<br/>and signing keys]
    J -->|Yes| L{Required role present?}
    L -->|No| M[Add role in Azure AD]
    
    style C fill:#dc3545,color:#fff
    style E fill:#dc3545,color:#fff
    style G fill:#ffc107,color:#000
    style I fill:#ffc107,color:#000
    style K fill:#dc3545,color:#fff
    style M fill:#ffc107,color:#000
```

**Debug Commands:**

```bash
# Decode token to inspect claims (use jwt.io or)
echo "YOUR_TOKEN_HERE" | cut -d'.' -f2 | base64 -d | jq

# Check specific claims
# aud: should match backend client ID
# iss: should match expected issuer
# exp: should be in future (unix timestamp)
# roles: should contain required role
```

#### Issue 4: SSO Not Working Between Apps

**Checklist:**

```mermaid
graph TD
    A[SSO Not Working] --> B{Same browser?}
    B -->|No| C[❌ SSO requires same browser]
    B -->|Yes| D{Both use localStorage?}
    D -->|No| E[✅ Configure localStorage]
    D -->|Yes| F{Same Azure tenant?}
    F -->|No| G[❌ Must be same tenant]
    F -->|Yes| H{Using ssoSilent?}
    H -->|No| I[✅ Use ssoSilent method]
    H -->|Yes| J{Check MSAL logs}
    J --> K[Enable verbose logging]
    
    style C fill:#dc3545,color:#fff
    style E fill:#28a745,color:#fff
    style G fill:#dc3545,color:#fff
    style I fill:#28a745,color:#fff
```

**Enable Debug Logging:**

```javascript
const msalConfig = {
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return;
        console.log(`[MSAL][${LogLevel[level]}]`, message);
      },
      logLevel: LogLevel.Verbose, // Most detailed
      piiLoggingEnabled: false,
    },
  },
}
```

#### Issue 5: "AADSTS50011: Reply URL mismatch"

**Cause:** Redirect URI not registered in Azure AD.

**Solution:**
1. Go to Azure Portal → App Registrations → Your App
2. Click "Authentication"
3. Under "Single-page application", add redirect URIs:
   - `http://localhost:5173`
   - `http://localhost:5173/`
   - Your production URL
4. Click "Save"
5. Wait 5-10 minutes for propagation

---

## Best Practices

### Security Best Practices

#### 1. Never Log Tokens

```javascript
// ❌ BAD - Exposes tokens in logs
console.log('Token:', accessToken);

// ✅ GOOD - Log metadata only
console.log('Token acquired, expires:', expiresOn);
```

#### 2. Always Validate Tokens Server-Side

```python
# ✅ GOOD - Validate everything
payload = jwt.decode(
    token,
    signing_key.key,
    algorithms=["RS256"],
    audience=config.client_id,  # Validate audience
    issuer=config.issuer,       # Validate issuer
    options={
        "verify_signature": True,  # Verify signature
        "verify_exp": True,        # Check expiration
        "verify_aud": True,
        "verify_iss": True,
    }
)

# ❌ BAD - Trusting without verification
payload = jwt.decode(token, options={"verify_signature": False})
```

#### 3. Use HTTPS in Production

```javascript
// ✅ GOOD - Production config
const msalConfig = {
  auth: {
    redirectUri: 'https://yourdomain.com',  // HTTPS
  }
}

// ⚠️ OK for development only
const msalConfig = {
  auth: {
    redirectUri: 'http://localhost:5173',  // HTTP OK locally
  }
}
```

#### 4. Implement Token Refresh

```javascript
// ✅ GOOD - Automatic refresh
async function callAPI() {
  try {
    const token = await msalInstance.acquireTokenSilent(tokenRequest);
    return fetch(url, {
      headers: { 'Authorization': `Bearer ${token.accessToken}` }
    });
  } catch (error) {
    if (error instanceof InteractionRequiredAuthError) {
      // Fallback to interactive
      const token = await msalInstance.acquireTokenPopup(tokenRequest);
      return fetch(url, {
        headers: { 'Authorization': `Bearer ${token.accessToken}` }
      });
    }
    throw error;
  }
}

// ❌ BAD - No refresh handling
const token = localStorage.getItem('token');
fetch(url, { headers: { 'Authorization': `Bearer ${token}` } });
```

### Performance Best Practices

#### 1. Cache JWKS Keys

```python
from functools import lru_cache

@lru_cache()
def get_jwks_client():
    """Cached JWKS client - fetches keys once"""
    return PyJWKClient(jwks_uri)

# ✅ Keys cached, not fetched on every request
```

#### 2. Minimize Token Requests

```javascript
// ✅ GOOD - Request token once, reuse
const token = await acquireTokenSilent();
await Promise.all([
  fetch('/api/users', { headers: { Authorization: `Bearer ${token}` } }),
  fetch('/api/data', { headers: { Authorization: `Bearer ${token}` } }),
]);

// ❌ BAD - Multiple token requests
await fetch('/api/users', { 
  headers: { Authorization: `Bearer ${await acquireTokenSilent()}` } 
});
await fetch('/api/data', { 
  headers: { Authorization: `Bearer ${await acquireTokenSilent()}` } 
});
```

#### 3. Lazy Load MSAL

```javascript
// ✅ GOOD - Initialize MSAL asynchronously
const initAndRender = async () => {
  await msalInstance.initialize();
  await msalInstance.handleRedirectPromise();
  render(<App />);
};

initAndRender();

// ❌ BAD - Blocking initialization
const msalInstance = new PublicClientApplication(msalConfig);
msalInstance.initialize(); // Blocks
render(<App />); // Delayed
```

### Code Organization Best Practices

#### 1. Centralize Auth Logic

```
src/
├── auth/
│   ├── authConfig.js       # MSAL configuration
│   ├── msalInstance.js     # Singleton instance
│   ├── hooks/
│   │   ├── useAuth.js      # Auth state hook
│   │   ├── useLogout.js    # Logout hook
│   │   └── useToken.js     # Token acquisition hook
│   └── utils/
│       ├── logout.js       # Logout utilities
│       └── storage.js      # Cache helpers
```

#### 2. Use Custom Hooks

```javascript
// useAuth.js
export function useAuth() {
  const { instance, accounts, inProgress } = useMsal();
  
  return {
    isAuthenticated: accounts.length > 0,
    user: accounts[0],
    isLoading: inProgress !== 'none',
    login: () => instance.loginPopup(),
    logout: () => instance.logoutPopup(),
  };
}

// Component usage
function MyComponent() {
  const { isAuthenticated, user, login } = useAuth();
  
  if (!isAuthenticated) {
    return <button onClick={login}>Sign In</button>;
  }
  
  return <div>Welcome, {user.name}!</div>;
}
```

#### 3. Environment-Specific Configuration

```javascript
// config/auth.config.js
const configs = {
  development: {
    clientId: 'dev-client-id',
    authority: 'https://login.microsoftonline.com/dev-tenant-id',
    redirectUri: 'http://localhost:5173',
  },
  production: {
    clientId: 'prod-client-id',
    authority: 'https://login.microsoftonline.com/prod-tenant-id',
    redirectUri: 'https://yourdomain.com',
  },
};

export const msalConfig = {
  auth: configs[import.meta.env.MODE],
  cache: {
    cacheLocation: 'localStorage',
    storeAuthStateInCookie: true,
  },
};
```

### Testing Best Practices

#### 1. Mock MSAL in Tests

```javascript
// __mocks__/@azure/msal-react.js
export const useMsal = jest.fn(() => ({
  instance: {
    loginPopup: jest.fn(),
    logoutPopup: jest.fn(),
    acquireTokenSilent: jest.fn(),
  },
  accounts: [{
    username: 'test@example.com',
    name: 'Test User',
  }],
  inProgress: 'none',
}));

export const MsalProvider = ({ children }) => children;
```

#### 2. Test Different Auth States

```javascript
describe('ChatHeader', () => {
  it('shows user name when authenticated', () => {
    useMsal.mockReturnValue({
      accounts: [{ name: 'John Doe' }],
      inProgress: 'none',
    });
    
    const { getByText } = render(<ChatHeader />);
    expect(getByText('John Doe')).toBeInTheDocument();
  });
  
  it('shows login button when not authenticated', () => {
    useMsal.mockReturnValue({
      accounts: [],
      inProgress: 'none',
    });
    
    const { getByText } = render(<ChatHeader />);
    expect(getByText('Sign In')).toBeInTheDocument();
  });
});
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **OAuth 2.0** | Industry-standard protocol for authorization |
| **OpenID Connect** | Identity layer on top of OAuth 2.0 |
| **MSAL** | Microsoft Authentication Library |
| **JWT** | JSON Web Token - compact, URL-safe means of representing claims |
| **PKCE** | Proof Key for Code Exchange - security extension for OAuth 2.0 |
| **SPA** | Single Page Application |
| **SSO** | Single Sign-On |
| **JWKS** | JSON Web Key Set - public keys for verifying JWTs |
| **Claims** | Statements about an entity (user) in a JWT |
| **Scope** | Permission to access specific resources |
| **Role** | Set of permissions assigned to a user |
| **Tenant** | Azure AD organization instance |
| **Authority** | Authentication endpoint URL |
| **Audience** | Intended recipient of a token |
| **Issuer** | Entity that created and signed a token |

### Quick Reference: MSAL Methods

```javascript
// Authentication
await instance.loginPopup(request)
await instance.loginRedirect(request)
await instance.logout Popup()
await instance.logoutRedirect()

// Token Acquisition
await instance.acquireTokenSilent(request)
await instance.acquireTokenPopup(request)
await instance.acquireTokenRedirect(request)
await instance.ssoSilent(request)

// Account Management
instance.getAllAccounts()
instance.getActiveAccount()
instance.setActiveAccount(account)
instance.getAccountByUsername(username)

// Event Handling
instance.addEventCallback(callback)
instance.removeEventCallback(callbackId)
instance.enableAccountStorageEvents()

// Initialization
await instance.initialize()
await instance.handleRedirectPromise()
```

### Quick Reference: JWT Claims

```json
{
  "aud": "a8a16485-0827-46c6-b3e0-91fca5966341",
  "iss": "https://login.microsoftonline.com/{tenant-id}/v2.0",
  "iat": 1698765432,
  "nbf": 1698765432,
  "exp": 1698769032,
  "sub": "ABC123...",
  "oid": "user-object-id",
  "tid": "tenant-id",
  "name": "John Doe",
  "preferred_username": "john.doe@example.com",
  "email": "john.doe@example.com",
  "roles": ["rag_chat_user"],
  "scp": "access_as_user User.Read",
  "ver": "2.0"
}
```

### Resources

- [Microsoft Identity Platform Documentation](https://learn.microsoft.com/en-us/azure/active-directory/develop/)
- [MSAL.js Documentation](https://github.com/AzureAD/microsoft-authentication-library-for-js)
- [OAuth 2.0 Specification](https://oauth.net/2/)
- [JWT.io Token Debugger](https://jwt.io)
- [PKCE RFC 7636](https://tools.ietf.org/html/rfc7636)

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-15 | Initial documentation |
| 2.0 | 2025-10-20 | Added Mermaid diagrams, cross-app SSO, comprehensive troubleshooting |

---

**End of Document**

For questions or issues, please refer to the project's GitHub repository or contact the development team.
