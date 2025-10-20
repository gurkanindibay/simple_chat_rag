import { ConfigCard } from './ConfigCard';
import { PDFUploadCard } from './PDFUploadCard';
import { IngestedPDFsCard } from './IngestedPDFsCard';
import { StatsCard } from './StatsCard';
import { DeleteButton } from './DeleteButton';
import { useMsal } from '@azure/msal-react';

export function Sidebar({ config, ingested, stats, onRefresh }) {
  const { instance } = useMsal();

  const handleSignOut = () => {
    // Set global logout marker so other same-host apps (different port) can detect it.
    try {
      const logoutMarker = Date.now().toString();
      localStorage.setItem('msal_global_logout', logoutMarker);
      localStorage.setItem('msal_global_logout_processed', logoutMarker);
      try {
        document.cookie = `msal_global_logout=${logoutMarker}; path=/; max-age=300`;
      } catch (cookieError) {
        console.warn('Unable to set msal_global_logout cookie during logout.', cookieError);
      }

      // Clear MSAL-related localStorage entries (except the global marker keys)
      const keysToRemove = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (key.includes('msal') || key.includes('login.windows')) && key !== 'msal_global_logout' && key !== 'msal_global_logout_processed') {
          keysToRemove.push(key);
        }
      }
      keysToRemove.forEach(key => localStorage.removeItem(key));

      // Try to clear MSAL-related cookies as well
      try {
        document.cookie.split(';').forEach(cookie => {
          const cookieName = cookie.split('=')[0].trim();
          if ((cookieName.includes('msal') || cookieName.includes('login.windows')) && cookieName !== 'msal_global_logout') {
            document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
          }
        });
      } catch (cookieClearError) {
        // Best-effort only
      }
    } catch (e) {
      console.warn('Error while setting global logout marker:', e);
    }

    // Use redirect for sign-out to ensure full cleanup in SPA scenarios
    instance.logoutRedirect({ postLogoutRedirectUri: '/' }).catch((e) => console.error('Logout redirect failed', e));
  };

  const handleSSOShowcase = () => {
    // Open the standalone SSO showcase SPA (independent application on port 8001)
    window.open('http://localhost:8001', '_blank');
  };
  
  const handleConfigChange = (updatedConfig) => {
    console.log('Config updated:', updatedConfig);
    // Refresh all data to get latest stats with new config
    if (onRefresh) {
      onRefresh();
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-cards">
        <ConfigCard config={config} onConfigChange={handleConfigChange} />
        <PDFUploadCard onFileUploaded={onRefresh} />
        <IngestedPDFsCard ingested={ingested} />
        <StatsCard stats={stats} />
      </div>
      <div className="sidebar-actions">
        <DeleteButton onDeleted={onRefresh} />
        <button 
          className="sso-showcase-button"
          onClick={handleSSOShowcase}
          title="View SSO Showcase (Standalone SPA)"
        >
          <i className="fas fa-shield-alt"></i>
          <span>SSO Showcase</span>
        </button>
        <button className="signout-button" onClick={handleSignOut} title="Sign out">
          <i className="fas fa-sign-out-alt"></i>
          <span>Sign out</span>
        </button>
      </div>
    </div>
  );
}
