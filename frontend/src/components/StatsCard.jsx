export function StatsCard({ stats }) {
  if (!stats || Object.keys(stats).length === 0) {
    return (
      <div className="card">
        <h3>
          <i className="fas fa-database card-icon"></i> Vector DB Stats
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
        <i className="fas fa-database card-icon"></i> Vector DB Stats
      </h3>
      <table className="stats-table">
        <thead>
          <tr>
            <th>Table</th>
            <th>Count</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(stats).map(([table, count]) => (
            <tr key={table}>
              <td>{table}</td>
              <td className="stat-count">{count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
