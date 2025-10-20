import { useEffect, useState } from 'react';
import { useIsAuthenticated, useMsal } from '@azure/msal-react';
import { ChatHeader } from './components/ChatHeader';
import { MessagesList } from './components/MessagesList';
import { ChatInput } from './components/ChatInput';
import { Sidebar } from './components/Sidebar';
import { Login } from './components/Login';
import { DebugPanel } from './components/DebugPanel';
import { useChatAPI } from './hooks/useChatAPI';
import { useAppData } from './hooks/useAppData';
import './styles/main.css';

function App() {
  const { instance, accounts, inProgress } = useMsal();
  const isAuthenticatedHook = useIsAuthenticated();
  const [isInitializing, setIsInitializing] = useState(true);
  
  // Fallback: if MSAL already has accounts in cache, consider the user authenticated
  const isAuthenticated = isAuthenticatedHook || (accounts && accounts.length > 0);
  const { messages, setMessages, loading, sendMessage } = useChatAPI();
  const { config, ingested, stats, loading: dataLoading, loadData } = useAppData();

  console.log('[App] Render - isAuthenticatedHook:', isAuthenticatedHook, 'accounts:', accounts?.length, 'isAuthenticated:', isAuthenticated, 'inProgress:', inProgress, 'isInitializing:', isInitializing);

  // Wait for MSAL to finish initializing before showing login page
  useEffect(() => {
    // Give MSAL a moment to load accounts from cache
    const timer = setTimeout(() => {
      setIsInitializing(false);
    }, 500);
    
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      // Clear any stale logout flags after successful authentication to prevent immediate logout
      localStorage.removeItem('msal_global_logout');
      localStorage.removeItem('msal_global_logout_processed');
      try {
        document.cookie = 'msal_global_logout=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
      } catch (cookieError) {
        console.warn('Unable to clear msal_global_logout cookie.', cookieError);
      }

      // Check for global logout flag
      const checkGlobalLogout = () => {
        const processedKey = 'msal_global_logout_processed';
        const processedValue = localStorage.getItem(processedKey);
        const lastProcessed = processedValue ? parseInt(processedValue, 10) : 0;

        // Check localStorage for global logout flag (works across same browser)
        const logoutFlag = localStorage.getItem('msal_global_logout');
        if (logoutFlag) {
          // Check if it's recent (within last 5 minutes)
          const logoutTime = parseInt(logoutFlag, 10);
          const now = Date.now();
          if (!Number.isNaN(logoutTime) && now - logoutTime < 300000 && logoutTime > lastProcessed) { // 5 minutes
            console.log('Global logout detected, logging out...');
            // Clear the flag
            localStorage.removeItem('msal_global_logout');
            localStorage.setItem(processedKey, logoutTime.toString());
            // Force logout
            instance.logoutRedirect();
            return;
          } else {
            // Flag is old, remove it
            localStorage.removeItem('msal_global_logout');
          }
        }
        
        // Also check cookie as fallback
        const cookies = document.cookie ? document.cookie.split(';') : [];
        const logoutCookie = cookies.find(cookie => cookie.trim().startsWith('msal_global_logout='));
        if (logoutCookie) {
          const cookieValue = parseInt(logoutCookie.split('=')[1], 10);
          if (!Number.isNaN(cookieValue) && cookieValue > lastProcessed) {
            console.log('Global logout detected via cookie, logging out...');
            localStorage.setItem(processedKey, cookieValue.toString());
            // Force logout
            instance.logoutRedirect();
            return;
          }
        }
      };
      
      // Check immediately and then every 5 seconds
      checkGlobalLogout();
      const logoutCheckInterval = setInterval(checkGlobalLogout, 5000);
      
      console.log('App mounted, loading initial data...');
      loadData();
      
      const interval = setInterval(() => {
        console.log('Refreshing data...');
        loadData();
      }, 5000);
      
      return () => {
        clearInterval(interval);
        clearInterval(logoutCheckInterval);
      };
    }
  }, [loadData, isAuthenticated, instance]);

  const handleMessageSent = async (message) => {
    console.log('Message sent:', message);
    setMessages(prev => [...prev, { role: 'user', text: message }]);
    try {
      await sendMessage(message);
    } catch (err) {
      console.error('Error sending message:', err);
    }
  };

  const handleRefresh = () => {
    console.log('Manual refresh triggered');
    loadData();
  };

  // Show loading state while MSAL is initializing to prevent flash of login page
  if (isInitializing || inProgress === 'startup' || inProgress === 'handleRedirect') {
    return (
      <div className="container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '10px' }}>Loading...</div>
          <div style={{ fontSize: '14px', color: '#666' }}>Checking authentication status</div>
        </div>
      </div>
    );
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <Login />;
  }

  // Show main app if authenticated
  return (
    <div className="container">
      <div className="chat-section">
        <ChatHeader userAccount={accounts[0]} />
        <MessagesList messages={messages} />
        <ChatInput 
          onMessageSent={handleMessageSent} 
          loading={loading}
        />
        <DebugPanel />
      </div>
      <Sidebar
        config={config}
        ingested={ingested}
        stats={stats}
        onRefresh={handleRefresh}
      />
    </div>
  );
}

export default App;
