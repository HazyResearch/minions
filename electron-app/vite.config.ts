// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  root: '.',            // 👈 make sure Vite uses root index.html
  base: './',           // 👈 fix asset paths for Electron
  build: {
    outDir: 'dist',     // 👈 make sure it outputs to dist/
    emptyOutDir: true,
  },
  server: {
    port: 5173,
  },
  // css: {
  //   postcss: './postcss.config.cjs'
  // }
});
