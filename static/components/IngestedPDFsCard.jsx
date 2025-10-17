const IngestedPDFsCard = ({ ingested }) => {
  if (!ingested) {
    return (
      <div className="card">
        <h3>
          <i className="fas fa-file-pdf card-icon"></i>
          Ingested PDFs
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
        <i className="fas fa-file-pdf card-icon"></i>
        Ingested PDFs ({ingested.length})
      </h3>
      {ingested.length === 0 ? (
        <div className="empty-state">
          <i className="fas fa-inbox"></i>
          <p>No PDFs ingested yet</p>
        </div>
      ) : (
        <div className="pdf-list">
          {ingested.map((item, idx) => (
            <div key={idx} className="pdf-item">
              <i className="fas fa-file-pdf pdf-icon"></i>
              <div className="pdf-info">
                <div className="pdf-name">{item.filename}</div>
                <div className="pdf-time">
                  {new Date(item.timestamp).toLocaleString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
