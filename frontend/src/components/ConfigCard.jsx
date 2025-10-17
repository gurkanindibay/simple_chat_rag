export function ConfigCard({ config }) {
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

  return (
    <div className="card">
      <h3>
        <i className="fas fa-cog card-icon"></i> Configuration
      </h3>
      <div className="config-item">
        <span className="config-label">LLM:</span>
        <span className="provider-badge">{config.LLM_PROVIDER || 'N/A'}</span>
      </div>
      <div className="config-item">
        <span className="config-label">Embeddings:</span>
        <span className="provider-badge">{config.EMBEDDING_PROVIDER || 'N/A'}</span>
      </div>
      <div className="config-item">
        <span className="config-label">Vector DB:</span>
        <span className="provider-badge">pgvector</span>
      </div>
    </div>
  );
}
