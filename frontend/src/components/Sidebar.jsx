import { ConfigCard } from './ConfigCard';
import { IngestedPDFsCard } from './IngestedPDFsCard';
import { StatsCard } from './StatsCard';
import { DeleteButton } from './DeleteButton';

export function Sidebar({ config, ingested, stats, onRefresh }) {
  return (
    <div className="sidebar">
      <div className="sidebar-cards">
        <ConfigCard config={config} />
        <IngestedPDFsCard ingested={ingested} />
        <StatsCard stats={stats} />
      </div>
      <div className="sidebar-actions">
        <DeleteButton onDeleted={onRefresh} />
      </div>
    </div>
  );
}
