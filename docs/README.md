# `docs/` — Standalone GitHub Pages build

This directory contains a **separate, standalone variant** of the app deployed
to GitHub Pages. It does **not** use the FastAPI backend in `../backend/`.

| File | Purpose |
|------|---------|
| `index.html` | Self-contained app — talks directly to Binance/yFinance from the browser, has its own JS TA engine, scanner, AI chat, and on-device NN candle predictor |
| `DOCUMENTATION.md` | Long-form documentation for the standalone build |
| `.nojekyll` | Disables Jekyll processing on GitHub Pages |

The primary app (driven by the Python backend) lives in `../frontend/`.
These two builds are intentionally separate; bug fixes do not propagate
automatically. If you only run one, prefer the backend-driven app — the
standalone build is for the GitHub Pages demo only.

The standalone build relies on the user running Ollama locally for AI chat
features; see `../start_ollama.bat`.
