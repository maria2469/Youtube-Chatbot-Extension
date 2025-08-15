import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  build: {
    outDir: path.resolve(__dirname, '../extension'),
    rollupOptions: {
      input: {
        popup: "index.html"
      },
    },
  },
})
