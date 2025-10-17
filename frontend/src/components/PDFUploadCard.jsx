import { useState, useRef } from 'react';
import { apiClient } from '../api/client';

export function PDFUploadCard({ onFileUploaded }) {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null);
  const fileInputRef = useRef(null);

  const showMessage = (msg, isError = false) => {
    setMessage({ text: msg, isError });
    setTimeout(() => setMessage(null), 3000);
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showMessage('✗ Please select a PDF file', true);
      return;
    }

    setUploading(true);

    try {
      await apiClient.ingestPDF(file);
      showMessage(`✓ ${file.name} uploaded successfully!`);
      
      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      // Notify parent to refresh data
      if (onFileUploaded) {
        onFileUploaded();
      }
    } catch (err) {
      console.error('Error uploading file:', err);
      showMessage(`✗ Failed to upload: ${err.message}`, true);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="card">
      <h3>
        <i className="fas fa-file-pdf card-icon"></i> Upload PDF
      </h3>
      
      {message && (
        <div className={`upload-message ${message.isError ? 'error' : 'success'}`}>
          {message.text}
        </div>
      )}

      <div className="upload-area">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".pdf"
          style={{ display: 'none' }}
          disabled={uploading}
        />
        
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn-upload"
          disabled={uploading}
          type="button"
        >
          {uploading ? (
            <>
              <i className="fas fa-spinner fa-spin"></i>
              <span>Uploading...</span>
            </>
          ) : (
            <>
              <i className="fas fa-cloud-upload-alt"></i>
              <span>Choose PDF File</span>
            </>
          )}
        </button>
        
        <p className="upload-hint">
          Upload a PDF to add to the knowledge base.
          <br />
          It will be processed using the <strong>Embedding Provider</strong> below.
        </p>
      </div>
    </div>
  );
}
