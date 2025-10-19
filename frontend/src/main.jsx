import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { PublicClientApplication } from '@azure/msal-browser'
import { MsalProvider } from '@azure/msal-react'
import { msalConfig } from './authConfig'
import { setMsalInstance, exposeMsalToWindow } from './api/client'
import '@fortawesome/fontawesome-free/css/all.min.css'
import './index.css'
import App from './App.jsx'

// Initialize MSAL instance
const msalInstance = new PublicClientApplication(msalConfig);

// Set the MSAL instance in the API client
setMsalInstance(msalInstance);
// Also expose msal instance on window for quick debugging (debugAuth helper available)
exposeMsalToWindow();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <MsalProvider instance={msalInstance}>
      <App />
    </MsalProvider>
  </StrictMode>,
)
