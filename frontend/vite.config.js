import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/config': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ingest': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ingestion-status': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/embeddings': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    }
  }
})
