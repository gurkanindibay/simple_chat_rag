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
    clientId: import.meta.env.VITE_AZURE_CLIENT_ID || '',
    authority: `https://login.microsoftonline.com/${import.meta.env.VITE_AZURE_TENANT_ID || 'common'}`,
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
  scopes: ['User.Read'], // Basic user profile read
};

/**
 * Scopes for token request (accessing your backend API)
 */
export const tokenRequest = {
  scopes: [
    `api://${import.meta.env.VITE_AZURE_CLIENT_ID}/access_as_user`,
    'User.Read'
  ],
};

/**
 * Graph API endpoint for getting user profile
 */
export const graphConfig = {
  graphMeEndpoint: 'https://graph.microsoft.com/v1.0/me',
};
