# Fuzzy Studio

A visual Fuzzy Inference System (FIS) editor and simulation environment. It includes a drag-and-drop rule builder, simulation steps inspector, and physical problem presets.

## Setup Instructions

This project is designed to be fully portable and runs entirely offline without needing Node.js or any internet connection after the first setup.

1. Install Python 3.9+
2. Clone the repository **with its submodule** (the `fuzzy_torch` backend lives in a separate repository):
   ```bash
   git clone --recursive https://github.com/mdrs-thiago/fuzzy-interface.git
   ```
   If you already cloned without `--recursive`, run `git submodule update --init` inside the project folder. An empty `fuzzy_torch/` folder means this step is missing.
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   - Double-click `run_app.bat` (Windows)
   - OR run `python -m uvicorn api:app --host 0.0.0.0 --port 8000`

5. Open your browser and go to `http://localhost:8000`

For development, add `--reload` to the uvicorn command to restart on code changes.

## Hosting

- Run a **single worker process** (the default for `uvicorn api:app`). Each visitor's FIS is kept in that process's memory, keyed by a session cookie, so visitors don't see or overwrite each other's systems. Multiple workers would each hold different sessions.
- Sessions are in memory only: they are cleared when the server restarts and after `FIS_SESSION_TTL` seconds of inactivity. Users can keep their work with the Export button.
- Behind an HTTPS reverse proxy, start uvicorn with `--proxy-headers --forwarded-allow-ips="*"` (or your proxy's IP) so the session cookie is marked secure.

Optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIS_SESSION_TTL` | `14400` | Seconds of inactivity before a session is discarded |
| `FIS_MAX_SESSIONS` | `1000` | Maximum concurrent sessions; the least recently used is dropped beyond this |
| `FIS_MAX_CSV_BYTES` | `5242880` | Largest CSV accepted by the Wang-Mendel extractor |
| `FIS_ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins allowed to call the API cross-site. Not needed for the bundled UI |

## Editing the UI

The browser loads the compiled `ui/app.js` and `ui/app.css`. After changing `ui/app.jsx` or `ui/styles.css`, rebuild them with Node.js 18+:

```bash
npm install
npm run build
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest test_api.py
```

## Features

- **Portable Web UI:** Built with React and Tailwind; the compiled `ui/app.js` and `ui/app.css` are committed, so running the app needs no Node.js.
- **Fuzzy Torch:** A PyTorch-based fuzzy logic backend that supports batch inference and differentiability.
- **Wang-Mendel Data-Driven FIS:** Extract fuzzy rules from raw numerical data pairs.
- **Lecture Exercises:** Pre-loaded presets like the "Tipping Problem" and "Broken Tank" for interactive classroom learning.
