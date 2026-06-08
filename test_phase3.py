import requests
import time
import subprocess
import sys

# Ensure server is running (it should be from previous steps, but let's assume it is)
# If not, we might need to start it. Let's try to connect first.

BASE_URL = "http://127.0.0.1:8008"

def test():
    print("Testing Phase 3 Features...")
    
    # 1. Reset
    requests.post(f"{BASE_URL}/reset")

    # 2. Create Variables with Roles
    res = requests.post(f"{BASE_URL}/variable", json={"name": "Service", "min_val": 0, "max_val": 10, "role": "input"})
    assert res.status_code == 200
    
    res = requests.post(f"{BASE_URL}/variable", json={"name": "Tip", "min_val": 0, "max_val": 30, "role": "output"})
    assert res.status_code == 200

    # 3. Add Terms
    requests.post(f"{BASE_URL}/variable/Service/term", json={"name": "Good", "type": "tri", "params": [0, 5, 10]})
    requests.post(f"{BASE_URL}/variable/Tip/term", json={"name": "High", "type": "tri", "params": [15, 30, 30]})

    # 4. Add Rule
    res = requests.post(f"{BASE_URL}/rule", json={
        "antecedents": {"Service": "Good"},
        "consequent": {"Tip": "High"}
    })
    assert res.status_code == 200
    
    # 5. Build (Auto-detect output)
    res = requests.post(f"{BASE_URL}/build")
    if res.status_code != 200:
        print(f"Build Failed: {res.text}")
    assert res.status_code == 200
    print("Build Successful (Auto-detected Output)")

    # 6. Simulate
    res = requests.post(f"{BASE_URL}/simulate", json={"inputs": {"Service": 5.0}})
    assert res.status_code == 200
    print(f"Simulation Result: {res.json()}")

    # 7. Delete Rule
    res = requests.delete(f"{BASE_URL}/rule/0")
    assert res.status_code == 200
    print("Rule Deleted")

    # 8. Delete Term
    res = requests.delete(f"{BASE_URL}/variable/Service/term/Good")
    assert res.status_code == 200
    print("Term Deleted")
    
    # 9. Delete Variable
    res = requests.delete(f"{BASE_URL}/variable/Tip")
    assert res.status_code == 200
    print("Variable Deleted")

    print("ALL PHASE 3 TESTS PASSED")

if __name__ == "__main__":
    test()
