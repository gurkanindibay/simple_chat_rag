import { useState, useCallback } from 'react';
import { apiClient } from '../api/client';

export const useAppData = () => {
  const [config, setConfig] = useState(null);
  const [ingested, setIngested] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      console.log('Loading app data...');
      const [configData, ingestData, statsData] = await Promise.all([
        apiClient.getConfig().catch(e => { console.error('Config error:', e); throw e; }),
        apiClient.getIngestionStatus().catch(e => { console.error('Ingestion error:', e); throw e; }),
        apiClient.getEmbeddingsStatus().catch(e => { console.error('Stats error:', e); throw e; }),
      ]);
      
      console.log('App data loaded:', { configData, ingestData, statsData });
      setConfig(configData);
      setIngested(ingestData.ingested || []);
      setStats(statsData.tables || {});
    } catch (err) {
      console.error('Error loading app data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  return { config, ingested, stats, loading, loadData };
};
