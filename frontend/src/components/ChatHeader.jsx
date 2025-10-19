import { useMsal } from '@azure/msal-react';

export function ChatHeader({ userAccount }) {
  const { instance, accounts } = useMsal();

  // Prefer passed userAccount, otherwise use first account from MSAL
  const acct = userAccount || (accounts && accounts[0]) || null;

  // Derive a friendly display name from common fields
  const displayName = acct
    ? (acct.name || acct.username || (acct.idTokenClaims && (acct.idTokenClaims.name || acct.idTokenClaims.preferred_username)) || acct.homeAccountId)
    : null;

  const handleLogout = () => {
    instance.logoutPopup({
      postLogoutRedirectUri: "/",
    }).catch((error) => {
      console.error('Logout error:', error);
    });
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
            <button className="logout-button" onClick={handleLogout} title="Sign out">
              <i className="fas fa-sign-out-alt"></i>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
