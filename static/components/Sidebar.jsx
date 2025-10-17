const Sidebar = ({
  config,
  ingested,
  stats,
  onDelete,
  isDeleting,
  deleteResult,
}) => {
  return (
    <div className="sidebar">
      <ConfigCard config={config} />
      <IngestedPDFsCard ingested={ingested} />
      <StatsCard stats={stats} />
      <DeleteButton
        onDelete={onDelete}
        isDeleting={isDeleting}
        deleteResult={deleteResult}
      />
    </div>
  );
};
