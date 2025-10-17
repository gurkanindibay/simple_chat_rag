import { useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';

export function MessagesList({ messages }) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-messages">
      {messages.length === 0 ? (
        <div className="empty-state">
          <i className="fas fa-comments"></i>
          <p>Start a conversation by uploading a PDF or asking a question</p>
        </div>
      ) : (
        messages.map((msg, idx) => (
          <ChatMessage
            key={idx}
            message={msg.text}
            isUser={msg.role === 'user'}
            sources={msg.sources}
          />
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
