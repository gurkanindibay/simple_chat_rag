import { useState, useCallback } from 'react';
import { apiClient } from '../api/client';

export const useChatAPI = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (question) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.chat(question);
      setMessages(prev => [
        ...prev,
        { 
          role: 'bot', 
          text: response.answer || 'No response received',
          sources: response.sources || []
        }
      ]);
    } catch (err) {
      console.error('Chat error:', err);
      setError(err.message);
      setMessages(prev => [
        ...prev,
        { role: 'bot', text: `Error: ${err.message}` }
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  return { messages, setMessages, loading, error, sendMessage };
};
