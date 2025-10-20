import { ConfigCard } from './ConfigCard';
import { PDFUploadCard } from './PDFUploadCard';
import { IngestedPDFsCard } from './IngestedPDFsCard';
import { StatsCard } from './StatsCard';
import { DeleteButton } from './DeleteButton';
import { useLogout } from '../hooks/useLogout';

export function Sidebar({ config, ingested, stats, onRefresh }) {
  const { logout, isLoggingOut } = useLogout({ logoutType: 'redirect' });

  const handleSignOut = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Sign out error:', error);
    }
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
        <button className="signout-button" onClick={handleSignOut} disabled={isLoggingOut} title={isLoggingOut ? "Signing out..." : "Sign out"}>
          <i className={`fas ${isLoggingOut ? 'fa-spinner fa-spin' : 'fa-sign-out-alt'}`}></i>
          <span>{isLoggingOut ? 'Signing out...' : 'Sign out'}</span>
        </button>
      </div>
    </div>
  );
}
