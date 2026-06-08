# Fuzzy Studio

A visual Fuzzy Inference System (FIS) editor and simulation environment. It includes a drag-and-drop rule builder, simulation steps inspector, and physical problem presets.

## Setup Instructions

This project is designed to be fully portable and runs entirely offline without needing Node.js or any internet connection after the first setup.

1. Install Python 3.9+
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   - Double-click `run_app.bat` (Windows)
   - OR run `python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000`

4. Open your browser and go to `http://localhost:8000`

## Features

- **Portable Web UI:** Built with React but loaded directly in the browser via Babel standalone. No Webpack, no `npm install`.
- **Fuzzy Torch:** A PyTorch-based fuzzy logic backend that supports batch inference and differentiability.
- **Wang-Mendel Data-Driven FIS:** Extract fuzzy rules from raw numerical data pairs.
- **Lecture Exercises:** Pre-loaded presets like the "Tipping Problem" and "Broken Tank" for interactive classroom learning.
