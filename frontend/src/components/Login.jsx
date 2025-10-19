import { useMsal } from '@azure/msal-react';
import { loginRequest } from '../authConfig';
import './Login.css';

export const Login = () => {
  const { instance } = useMsal();

  const handleLogin = (loginType) => {
    if (loginType === 'popup') {
      instance.loginPopup(loginRequest)
        .catch((error) => {
          console.error('Login error:', error);
        });
    } else if (loginType === 'redirect') {
      instance.loginRedirect(loginRequest)
        .catch((error) => {
          console.error('Login error:', error);
        });
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1>RAG Chat Application</h1>
          <p>Please sign in with your Microsoft account</p>
        </div>
        
        <div className="login-content">
          <div className="login-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="64" height="64">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
          </div>
          
          <div className="login-buttons">
            <button 
              className="login-button login-button-primary" 
              onClick={() => handleLogin('popup')}
            >
              <svg className="microsoft-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 23 23">
                <path fill="#f35325" d="M0 0h11v11H0z"/>
                <path fill="#81bc06" d="M12 0h11v11H12z"/>
                <path fill="#05a6f0" d="M0 12h11v11H0z"/>
                <path fill="#ffba08" d="M12 12h11v11H12z"/>
              </svg>
              Sign in with Microsoft (Popup)
            </button>
            
            <button 
              className="login-button login-button-secondary" 
              onClick={() => handleLogin('redirect')}
            >
              <svg className="microsoft-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 23 23">
                <path fill="#f35325" d="M0 0h11v11H0z"/>
                <path fill="#81bc06" d="M12 0h11v11H12z"/>
                <path fill="#05a6f0" d="M0 12h11v11H0z"/>
                <path fill="#ffba08" d="M12 12h11v11H12z"/>
              </svg>
              Sign in with Microsoft (Redirect)
            </button>
          </div>

          <div className="login-footer">
            <p className="login-note">
              By signing in, you agree to use your organizational account
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
