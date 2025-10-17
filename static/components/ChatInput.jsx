const ChatInput = ({
  message,
  setMessage,
  onSendMessage,
  onFileSelected,
  isSending,
}) => {
  const fileInputRef = React.useRef(null);

  const handleSend = () => {
    if (message.trim()) {
      onSendMessage();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
    }
  };

  return (
    <div className="chat-input-area">
      <div className="input-wrapper">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about the PDFs..."
          style={{ flex: 1 }}
        />
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".pdf"
          style={{ display: 'none' }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          title="Upload PDF"
          style={{
            background: 'white',
            color: '#667eea',
            border: '1px solid #ddd',
          }}
        >
          <i className="fas fa-upload"></i>
        </button>
      </div>
      <button
        onClick={handleSend}
        disabled={isSending || !message.trim()}
        className="btn-primary"
      >
        {isSending ? (
          <>
            <i className="fas fa-spinner loading"></i>
          </>
        ) : (
          <>
            <i className="fas fa-paper-plane"></i>
          </>
        )}
      </button>
    </div>
  );
};
