import { useState } from 'react';
import { apiClient } from '../api/client';

export function DeleteButton({ onDeleted }) {
  const [deleting, setDeleting] = useState(false);
  const [result, setResult] = useState(null);

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete all embeddings? This cannot be undone.')) {
      return;
    }

    setDeleting(true);
    setResult(null);

    try {
      const response = await apiClient.deleteEmbeddings();
      setResult({ status: 'success', message: 'Embeddings deleted successfully' });
      onDeleted?.();
    } catch (err) {
      setResult({ status: 'error', message: err.message });
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="card">
      <button
        onClick={handleDelete}
        disabled={deleting}
        className="btn-danger"
        style={{ width: '100%' }}
      >
        <i className="fas fa-trash"></i> {deleting ? 'Deleting...' : 'Delete All Embeddings'}
      </button>
      {result && (
        <div className={`delete-result ${result.status}`}>
          {result.message}
        </div>
      )}
    </div>
  );
}
