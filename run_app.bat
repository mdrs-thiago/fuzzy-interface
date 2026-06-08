@echo off
echo Installing requirements...
pip install -r requirements.txt
echo Starting Fuzzy Inference System...
echo Open http://localhost:8000 in your browser.
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
pause
