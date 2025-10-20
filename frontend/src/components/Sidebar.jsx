import { ConfigCard } from './ConfigCard';
import { PDFUploadCard } from './PDFUploadCard';
import { IngestedPDFsCard } from './IngestedPDFsCard';
import { StatsCard } from './StatsCard';
import { DeleteButton } from './DeleteButton';
import { useMsal } from '@azure/msal-react';

export function Sidebar({ config, ingested, stats, onRefresh }) {
  const { instance } = useMsal();

  const handleSignOut = () => {
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
