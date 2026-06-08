import requests
import time
import subprocess
import sys

# Start the server in background
print("Starting API server...")
# Correcting to run via python direct execution which has uvicorn.run inside
proc = subprocess.Popen([sys.executable, "api.py"], cwd="e:/projects/fuzzy")
time.sleep(5) # Wait for startup

BASE_URL = "http://127.0.0.1:8000"

try:
    print("Testing API...")
    
    # 1. Check Root (UI)
    res = requests.get(BASE_URL + "/")
    assert res.status_code == 200, "UI not served"
    print("UI served successfully.")

    # 2. Create Variable
    res = requests.post(f"{BASE_URL}/variable", json={"name": "TestVar", "min_val": 0, "max_val": 10})
    assert res.status_code == 200
    
    res = requests.post(f"{BASE_URL}/variable", json={"name": "OutputVar", "min_val": 0, "max_val": 10})
    assert res.status_code == 200

    # 3. Add Terms
    res = requests.post(f"{BASE_URL}/variable/TestVar/term", json={"name": "Low", "type": "tri", "params": [0, 0, 5]})
    assert res.status_code == 200
    
    res = requests.post(f"{BASE_URL}/variable/OutputVar/term", json={"name": "High", "type": "tri", "params": [5, 10, 10]})
    assert res.status_code == 200

    # 4. Add Rule
    res = requests.post(f"{BASE_URL}/rule", json={
        "antecedents": {"TestVar": "Low"},
        "consequent": {"OutputVar": "High"}
    })
    assert res.status_code == 200

    # 5. Build
    res = requests.post(f"{BASE_URL}/build?output_var_name=OutputVar")
    assert res.status_code == 200

    # 6. Simulate
    res = requests.post(f"{BASE_URL}/simulate", json={"inputs": {"TestVar": 2.5}})
    assert res.status_code == 200
    data = res.json()
    print(f"Simulation Result: {data}")
    assert "output" in data
    assert len(data["rules_triggered"]) > 0

    print("ALL TESTS PASSED.")

except Exception as e:
    print(f"TEST FAILED: {e}")
finally:
    proc.terminate()
