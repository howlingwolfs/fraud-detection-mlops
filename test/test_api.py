import sys
import requests

# Set endpoint bases based on where your app is running
LOCAL_URL = "http://localhost:8000"
DOCKER_URL = "http://localhost:8000"  # Adjust host/port if Docker uses a different port (e.g. http://localhost:8080)

# Sample transaction payload matching the Pydantic schema
SAMPLE_PAYLOAD = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098,
    "V3": 2.5363467379698,
    "V4": 1.3781552242744,
    "V5": -0.3383207699425,
    "V6": 0.4623880443422,
    "V7": 0.2395985540608,
    "V8": 0.098697901261,
    "V9": 0.3637869696116,
    "V10": 0.0907941719789,
    "V11": -0.5515995332608,
    "V12": -0.6178008557616,
    "V13": -0.9913898472354,
    "V14": -0.3111693536999,
    "V15": 1.4681769720943,
    "V16": -0.4704005252594,
    "V17": 0.2079712419292,
    "V18": 0.025790580198,
    "V19": 0.4039929602557,
    "V20": 0.2514120982397,
    "V21": -0.0183067779441,
    "V22": 0.2778375755589,
    "V23": -0.110474010131,
    "V24": 0.066928074905,
    "V25": 0.1285393582735,
    "V26": -0.1891148438888,
    "V27": 0.1335583767403,
    "V28": -0.0265233482418,
    "Amount": 149.62
}


def test_api(base_url: str, label: str):
    print(f"\n==========================================")
    print(f" Testing API Target: [{label}] at {base_url}")
    print(f"==========================================")

    # 1. Test Health Endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"\n[GET /health]")
        print(f"  Status Code : {response.status_code}")
        print(f"  Response    : {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to connect to {base_url}/health")
        print(f"  Error: {e}")
        return False

    # 2. Test Single Prediction Endpoint
    try:
        response = requests.post(f"{base_url}/predict", json=SAMPLE_PAYLOAD, timeout=5)
        print(f"\n[POST /predict]")
        print(f"  Status Code : {response.status_code}")
        print(f"  Response    : {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to connect to {base_url}/predict")
        print(f"  Error: {e}")

    # 3. Test Batch Prediction Endpoint
    try:
        batch_payload = [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]
        response = requests.post(f"{base_url}/predict_batch", json=batch_payload, timeout=5)
        print(f"\n[POST /predict_batch]")
        print(f"  Status Code : {response.status_code}")
        print(f"  Response    : {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to connect to {base_url}/predict_batch")
        print(f"  Error: {e}")


if __name__ == "__main__":
    # Determine target from command line args if provided, else test local by default
    # Usage: python test_api.py [local|docker]
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "local"

    if target == "docker":
        test_api(DOCKER_URL, "Docker Container")
    else:
        test_api(LOCAL_URL, "Local Server")