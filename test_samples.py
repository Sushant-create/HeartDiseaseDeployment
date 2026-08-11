"""
test_samples.py
================
Simple smoke-test script that sends each record in test_samples.json to a
running instance of the API (local or the deployed Render URL) and prints
the prediction result. Useful for manually verifying the deployment works.

Usage:
    # against local dev server (python main.py, in another terminal)
    python test_samples.py

    # against the deployed Render app
    python test_samples.py https://your-app-name.onrender.com
"""

import json
import sys
import urllib.request
import urllib.error

DEFAULT_BASE_URL = "http://127.0.0.1:5000"


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")

    with open("test_samples.json") as f:
        samples = json.load(f)

    print(f"Testing API at: {base_url}\n")

    # health check first
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as resp:
            print("Health check:", resp.read().decode())
    except Exception as e:
        print(f"Health check failed: {e}")

    print()
    for sample in samples:
        payload = sample["input"]
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                print(f"[{sample['description']}]")
                print(f"  Input : {payload}")
                print(f"  Result: {result}\n")
        except urllib.error.HTTPError as e:
            print(f"[{sample['description']}] HTTP {e.code}: {e.read().decode()}\n")
        except Exception as e:
            print(f"[{sample['description']}] Request failed: {e}\n")


if __name__ == "__main__":
    main()
