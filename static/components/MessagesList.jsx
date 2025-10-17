const MessagesList = ({ messages, messagesEndRef }) => {
  return (
    <div className="chat-messages">
      {messages.length === 0 ? (
        <div className="empty-state">
          <i className="fas fa-comments"></i>
          <p>Start a conversation by typing a question or uploading a PDF</p>
        </div>
      ) : (
        messages.map((msg, idx) => (
          <ChatMessage
            key={idx}
            message={msg.text}
            isUser={msg.isUser}
            sources={msg.sources}
          />
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};
