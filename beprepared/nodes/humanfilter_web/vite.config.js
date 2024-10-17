import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import federation from '@originjs/vite-plugin-federation'

export default defineConfig({
  plugins: [
    vue(),
    federation({
      name: 'humanfilter',
      filename: 'remoteEntry.js',
      exposes: {
        'App': './App.vue',
      },
      shared: ['vue'],
    })
  ],
  build: {
    target: 'esnext',
    minify: false,
    outDir: path.resolve(__dirname, 'static'),
    emptyOutDir: true,
  },
})
