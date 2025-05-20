# minions-webgpu

> A lightweight WebGPU‑powered LLM demo with a Fastify backend, built on \[Vite].

## Prerequisites

Before you begin, make sure you have:

- **Node.js** (LTS v16 or above)
- **npm** (bundled with Node.js)
- A browser with **WebGPU** support (e.g. Chrome 113+, Edge Canary, or Safari Technology Preview)

## Installation

Install all dependencies:

```bash
npm install
```

> This will install:
>
> - `vite` (dev/build tool)
> - `@mlc-ai/web-llm` (WebGPU LLM runtime)
> - `fastify` (backend HTTP server)

## Available Scripts

In the project directory, you can run:

| Command           | Description                          |
| ----------------- | ------------------------------------ |
| `npm run dev`     | Start the Vite development server    |
| `npm run build`   | Build the app for production         |
| `npm run preview` | Preview the production build locally |

## Usage

1. Start the dev server:

   ```bash
   npm run dev
   ```

2. Open your browser and navigate to the URL printed in the console (usually `http://localhost:5173`).
3. Make sure your browser supports WebGPU to run the LLM in the client.
