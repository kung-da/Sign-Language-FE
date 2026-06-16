# VSL AI Translator Frontend

Academic/portfolio frontend demo for a real-time Vietnamese Sign Language AI translator.

## Run locally

```bash
npm install
npm run dev
```

## Create from scratch

```bash
npm create vite@latest vsl-ai-translator -- --template react-ts
cd vsl-ai-translator
npm install
npm install tailwindcss postcss autoprefixer framer-motion lucide-react
npx tailwindcss init -p
npm run dev
```

## Backend integration notes

Current predictions, vector search, FPS, latency, and save-new-sign behavior are mock data only. Replace `src/services/recognitionService.ts` and `src/services/vectorDbService.ts` with calls to a Python/FastAPI backend when the real AI pipeline is ready.
