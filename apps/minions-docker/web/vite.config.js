import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 8080,
    host: true,
    cors: true
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true
  }
})
