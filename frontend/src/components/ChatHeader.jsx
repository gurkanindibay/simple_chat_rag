import { useMsal } from '@azure/msal-react';

export function ChatHeader({ userAccount }) {
  const { instance } = useMsal();

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
      {userAccount && (
        <div className="chat-header-right">
          <div className="user-info">
            <span className="user-name">{userAccount.name || userAccount.username}</span>
            <button className="logout-button" onClick={handleLogout} title="Sign out">
              <i className="fas fa-sign-out-alt"></i>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
