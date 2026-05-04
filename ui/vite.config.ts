import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    allowedHosts: ["menadic-ciara-petiolar.ngrok-free.dev"],
    proxy: {
      '/telemetry': { target: 'http://127.0.0.1:8009', changeOrigin: true },
      '/assimilation_status': { target: 'http://127.0.0.1:8009', changeOrigin: true },
      '/comparison': { target: 'http://127.0.0.1:8009', changeOrigin: true },
      '/assimilate': { target: 'http://127.0.0.1:8009', changeOrigin: true },
      '/predict': { target: 'http://127.0.0.1:8009', changeOrigin: true },
      '/learn_signal': { target: 'http://127.0.0.1:8009', changeOrigin: true },
      '/encode': { target: 'http://127.0.0.1:8009', changeOrigin: true }
    }
  }
})
