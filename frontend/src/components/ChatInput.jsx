import { useRef } from 'react';
import { apiClient } from '../api/client';

export function ChatInput({ onMessageSent, loading, onFileUploaded }) {
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);

  const handleSend = async (e) => {
    e.preventDefault();
    const message = inputRef.current?.value.trim();
    
    if (!message) return;

    inputRef.current.value = '';
    await onMessageSent(message);
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      await apiClient.ingestPDF(file);
      onFileUploaded?.();
    } catch (err) {
      console.error('Error uploading file:', err);
    }
  };

  return (
    <div className="chat-input-area">
      <div className="input-wrapper">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".pdf"
          style={{ display: 'none' }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn-primary"
          title="Upload PDF"
          type="button"
        >
          <i className="fas fa-cloud-arrow-up"></i>
        </button>
        <input
          ref={inputRef}
          type="text"
          placeholder="Ask a question about the PDFs..."
          onKeyPress={(e) => e.key === 'Enter' && handleSend(e)}
          disabled={loading}
          style={{ flex: 1 }}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="btn-primary"
          type="submit"
          title={loading ? "Sending..." : "Send message"}
        >
          {loading ? (
            <i className="fas fa-spinner fa-spin"></i>
          ) : (
            <i className="fas fa-arrow-up"></i>
          )}
        </button>
      </div>
    </div>
  );
}
