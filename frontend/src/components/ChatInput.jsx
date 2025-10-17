import { useRef } from 'react';

export function ChatInput({ onMessageSent, loading }) {
  const inputRef = useRef(null);

  const handleSend = async (e) => {
    e.preventDefault();
    const message = inputRef.current?.value.trim();
    
    if (!message) return;

    inputRef.current.value = '';
    await onMessageSent(message);
  };

  return (
    <div className="chat-input-area">
      <div className="input-wrapper">
        <input
          ref={inputRef}
          type="text"
          placeholder="Ask a question about the PDFs..."
          onKeyPress={(e) => e.key === 'Enter' && handleSend(e)}
          disabled={loading}
          className="chat-input"
          style={{ color: '#333', backgroundColor: 'white' }}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="btn-primary btn-send"
          type="submit"
          title={loading ? "Sending..." : "Send message"}
        >
          {loading ? (
            <i className="fas fa-spinner fa-spin"></i>
          ) : (
            <i className="fas fa-paper-plane"></i>
          )}
        </button>
      </div>
    </div>
  );
}
