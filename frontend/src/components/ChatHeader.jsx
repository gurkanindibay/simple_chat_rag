import { useMsal } from '@azure/msal-react';
import { useLogout } from '../hooks/useLogout';

export function ChatHeader({ userAccount }) {
  const { accounts } = useMsal();
  const { logout, isLoggingOut } = useLogout({ logoutType: 'popup' });

  // Prefer passed userAccount, otherwise use first account from MSAL
  const acct = userAccount || (accounts && accounts[0]) || null;

  // Derive a friendly display name from common fields
  const displayName = acct
    ? (acct.name || acct.username || (acct.idTokenClaims && (acct.idTokenClaims.name || acct.idTokenClaims.preferred_username)) || acct.homeAccountId)
    : null;

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <div className="chat-header">
      <div className="chat-header-left">
        <i className="fas fa-robot"></i>
        <h1>RAG Chat Assistant</h1>
      </div>
      {acct && (
        <div className="chat-header-right">
          <div className="user-info">
            <span className="user-name">{displayName || 'User'}</span>
            <button 
              className="logout-button" 
              onClick={handleLogout} 
              disabled={isLoggingOut}
              title={isLoggingOut ? "Signing out..." : "Sign out"}
            >
              <i className={`fas ${isLoggingOut ? 'fa-spinner fa-spin' : 'fa-sign-out-alt'}`}></i>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
