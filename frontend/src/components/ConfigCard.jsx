import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';

export function ConfigCard({ config, onConfigChange }) {
  const [llmProvider, setLlmProvider] = useState('OPENAI');
  const [embeddingProvider, setEmbeddingProvider] = useState('OPENAI');
  const [updating, setUpdating] = useState(false);
  const [message, setMessage] = useState(null);

  // Initialize state from config when it loads
  useEffect(() => {
    if (config) {
      setLlmProvider(config.LLM_PROVIDER || 'OPENAI');
      setEmbeddingProvider(config.EMBEDDING_PROVIDER || 'OPENAI');
    }
  }, [config]);

  const showMessage = (msg, isError = false) => {
    setMessage({ text: msg, isError });
    setTimeout(() => setMessage(null), 3000);
  };

  if (!config) {
    return (
      <div className="card">
        <h3>
          <i className="fas fa-cog card-icon"></i> Configuration
        </h3>
        <div className="empty-state">
          <i className="fas fa-spinner loading"></i>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  const handleLlmChange = async (e) => {
    const newProvider = e.target.value;
    setUpdating(true);
    
    try {
      const response = await apiClient.updateConfig('LLM_PROVIDER', newProvider);
      setLlmProvider(newProvider);
      showMessage(`✓ LLM provider updated to ${newProvider}`);
      
      // Notify parent component
      if (onConfigChange) {
        onConfigChange(response.config);
      }
    } catch (error) {
      console.error('Failed to update LLM provider:', error);
      showMessage(`✗ Failed to update: ${error.message}`, true);
      // Revert to previous value
      setLlmProvider(config.LLM_PROVIDER || 'OPENAI');
    } finally {
      setUpdating(false);
    }
  };

  const handleEmbeddingChange = async (e) => {
    const newProvider = e.target.value;
    setUpdating(true);
    
    try {
      const response = await apiClient.updateConfig('EMBEDDING_PROVIDER', newProvider);
      setEmbeddingProvider(newProvider);
      showMessage(`✓ Embedding provider updated to ${newProvider}`);
      
      // Notify parent component
      if (onConfigChange) {
        onConfigChange(response.config);
      }
    } catch (error) {
      console.error('Failed to update Embedding provider:', error);
      showMessage(`✗ Failed to update: ${error.message}`, true);
      // Revert to previous value
      setEmbeddingProvider(config.EMBEDDING_PROVIDER || 'OPENAI');
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div className="card">
      <h3>
        <i className="fas fa-cog card-icon"></i> Configuration
      </h3>
      {message && (
        <div className={`config-message ${message.isError ? 'error' : 'success'}`}>
          {message.text}
        </div>
      )}
      <div className="config-item">
        <span className="config-label">LLM:</span>
        <select 
          className="provider-select" 
          value={llmProvider}
          onChange={handleLlmChange}
          disabled={updating}
        >
          <option value="OPENAI">OpenAI</option>
          <option value="LOCAL">Local</option>
        </select>
      </div>
      <div className="config-item">
        <span className="config-label">Embeddings:</span>
        <select 
          className="provider-select"
          value={embeddingProvider}
          onChange={handleEmbeddingChange}
          disabled={updating}
        >
          <option value="OPENAI">OpenAI</option>
          <option value="LOCAL">Local</option>
        </select>
      </div>
      <div className="config-item">
        <span className="config-label">Vector DB:</span>
        <span className="provider-badge">pgvector</span>
      </div>
    </div>
  );
}
