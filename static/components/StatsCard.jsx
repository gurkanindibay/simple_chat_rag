const StatsCard = ({ stats }) => {
  if (!stats) {
    return (
      <div className="card">
        <h3>
          <i className="fas fa-chart-bar card-icon"></i>
          Vector DB Statistics
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
        <i className="fas fa-chart-bar card-icon"></i>
        Vector DB Statistics
      </h3>
      <table className="stats-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((stat, idx) => (
            <tr key={idx}>
              <td>{stat.metric}</td>
              <td className="stat-count">{stat.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
