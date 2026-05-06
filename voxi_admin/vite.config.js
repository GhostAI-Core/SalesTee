import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/misses':    'http://localhost:8000',
      '/chat':      'http://localhost:8000',
      '/stats':     'http://localhost:8000',
      '/save':      'http://localhost:8000',
      '/qa':        'http://localhost:8000',
    },
  },
})
