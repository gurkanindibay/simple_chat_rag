import { ConfigCard } from './ConfigCard';
import { IngestedPDFsCard } from './IngestedPDFsCard';
import { StatsCard } from './StatsCard';
import { DeleteButton } from './DeleteButton';

export function Sidebar({ config, ingested, stats, onRefresh }) {
  return (
    <div className="sidebar">
      <ConfigCard config={config} />
      <IngestedPDFsCard ingested={ingested} />
      <StatsCard stats={stats} />
      <div className="delete-section">
        <DeleteButton onDeleted={onRefresh} />
      </div>
    </div>
  );
}
