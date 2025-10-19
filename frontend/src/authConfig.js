/**
 * Microsoft Entra (Azure AD) Authentication Configuration
 * 
 * To set up authentication:
 * 1. Go to Azure Portal > Azure Active Directory > App registrations
 * 2. Create a new app registration or use existing one
 * 3. Copy the Tenant ID and Client ID
 * 4. Add redirect URIs in "Authentication" settings:
 *    - http://localhost:5173 (for development)
 *    - Your production URL
 * 5. In "API permissions", add permissions as needed
 * 6. Create a .env file in frontend/ with these values
 */

export const msalConfig = {
  auth: {
  // Use explicit frontend environment variables for SPA client
  clientId: import.meta.env.VITE_AZURE_FRONTEND_CLIENT_ID || '',
  authority: `https://login.microsoftonline.com/${import.meta.env.VITE_AZURE_FRONTEND_TENANT_ID || 'common'}`,
    redirectUri: import.meta.env.VITE_REDIRECT_URI || window.location.origin,
  },
  cache: {
    cacheLocation: 'sessionStorage', // Use 'localStorage' for persistent login
    storeAuthStateInCookie: false, // Set to true if you have issues with IE11 or Edge
  },
};

/**
 * Scopes for login request
 * Add additional scopes as needed for your application
 */
export const loginRequest = {
  // Keep basic profile scopes for sign-in. MSAL will request tokens using tokenRequest when calling APIs.
  scopes: ['openid', 'profile', 'User.Read'],
};

/**
 * Scopes for token request (accessing your backend API)
 * 
 * NOTE: Currently using User.Read only. To call a custom backend API:
 * 1. In Azure Portal, go to your backend app registration → Expose an API
 * 2. Set Application ID URI (e.g., api://<backend-client-id>)
 * 3. Add a scope (e.g., access_as_user)
 * 4. In frontend app registration → API permissions → Add the backend scope
 * 5. Grant admin consent
 * 6. Update this to: scopes: [`api://<backend-client-id>/access_as_user`, 'User.Read']
 */
// Build backend API scope from env. The frontend .env may set one of these:
// - VITE_AZURE_BACKEND_SCOPE (full scope URI, e.g. api://<id>/access_as_user)
// - VITE_AZURE_BACKEND_CLIENT_ID (we'll construct api://<clientId>/access_as_user)
const envBackendScope = import.meta.env.VITE_AZURE_BACKEND_SCOPE
const backendClientId = import.meta.env.VITE_AZURE_BACKEND_CLIENT_ID
const defaultBackendScope = envBackendScope || (backendClientId ? `api://${backendClientId}/access_as_user` : null)

export const tokenRequest = {
  // Request an access token for the backend API when present; otherwise fall back to Graph scope.
  scopes: defaultBackendScope ? [defaultBackendScope] : ['User.Read'],
};

/**
 * Graph API endpoint for getting user profile
 */
export const graphConfig = {
  graphMeEndpoint: 'https://graph.microsoft.com/v1.0/me',
};
