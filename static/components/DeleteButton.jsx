const DeleteButton = ({ onDelete, isDeleting, deleteResult }) => {
  return (
    <div className="card">
      <h3>
        <i className="fas fa-trash card-icon"></i>
        Delete Embeddings
      </h3>
      <button
        onClick={onDelete}
        disabled={isDeleting}
        className="btn-danger"
        style={{ width: '100%' }}
      >
        {isDeleting ? (
          <>
            <i className="fas fa-spinner loading"></i> Deleting...
          </>
        ) : (
          <>
            <i className="fas fa-trash"></i> Delete All Embeddings
          </>
        )}
      </button>
      {deleteResult && (
        <div className={`delete-result ${deleteResult.status}`}>
          {deleteResult.status === 'success' ? (
            <>
              <i className="fas fa-check"></i> {deleteResult.message}
            </>
          ) : (
            <>
              <i className="fas fa-exclamation-circle"></i> {deleteResult.message}
            </>
          )}
        </div>
      )}
    </div>
  );
};
