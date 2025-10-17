const ConfigCard = ({ config }) => {
  if (!config) {
    return (
      <div className="card">
        <h3>
          <i className="fas fa-cog card-icon"></i>
          Configuration
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
        <i className="fas fa-cog card-icon"></i>
        Configuration
      </h3>
      <div className="config-item">
        <span className="config-label">LLM Provider:</span>
        <span className="provider-badge">{config.llm_provider}</span>
      </div>
      <div className="config-item">
        <span className="config-label">Embedding Provider:</span>
        <span className="provider-badge">{config.embedding_provider}</span>
      </div>
      <div className="config-item">
        <span className="config-label">Vector DB:</span>
        <span className="provider-badge">pgvector</span>
      </div>
      <div className="config-item">
        <span className="config-label">Retrieved Docs (k):</span>
        <span className="provider-badge">{config.retrieval_k}</span>
      </div>
    </div>
  );
};
