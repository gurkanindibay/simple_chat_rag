import { useEffect } from 'react';
import { ChatHeader } from './components/ChatHeader';
import { MessagesList } from './components/MessagesList';
import { ChatInput } from './components/ChatInput';
import { Sidebar } from './components/Sidebar';
import { useChatAPI } from './hooks/useChatAPI';
import { useAppData } from './hooks/useAppData';
import './styles/main.css';

function App() {
  const { messages, setMessages, loading, sendMessage } = useChatAPI();
  const { config, ingested, stats, loading: dataLoading, loadData } = useAppData();

  useEffect(() => {
    console.log('App mounted, loading initial data...');
    loadData();
    
    const interval = setInterval(() => {
      console.log('Refreshing data...');
      loadData();
    }, 5000);
    
    return () => clearInterval(interval);
  }, [loadData]);

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

  return (
    <div className="container">
      <div className="chat-section">
        <ChatHeader />
        <MessagesList messages={messages} />
        <ChatInput 
          onMessageSent={handleMessageSent} 
          loading={loading}
          onFileUploaded={handleRefresh}
        />
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
