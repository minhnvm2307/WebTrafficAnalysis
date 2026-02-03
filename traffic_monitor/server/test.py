# Test ml_server functionalities
from traffic_monitor.server.ml_server import app
from fastapi.testclient import TestClient
import json
client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().get('status') == "running"

def test_forecast_endpoint():
    # Model needs 64 x 15-min intervals = 960 minutes of data
    # Generate 1000 minutes of data (1 per minute) which will resample to ~67 15-min intervals
    look_back = 1000  # minutes of data
    test_data = {
        "history": [
        ]
    }
    for i in range(look_back):
        hours = i // 60
        minutes = i % 60
        test_data["history"].append({
            "timestamp": f"2024-01-01T{hours:02d}:{minutes:02d}:00Z",
            "request_count": 100 + i % 50,  # Varying pattern
            "bytes": 5000 + i * 10  # Add bytes field
        })

    response = client.post("/forecast", json=test_data)
    print(response.json())
    assert response.status_code == 200
    json_response = response.json()
    assert "forecast" in json_response
    assert len(json_response["forecast"]) > 0
    assert "timestamp_range" in json_response
    assert len(json_response["timestamp_range"]) == len(json_response["forecast"])

def test_anomaly_detection_endpoint():
    test_data = {
        "history": [
            {"timestamp": "2024-01-01T00:00:00Z", "request_count": 100, "bytes": 5000},
            {"timestamp": "2024-01-01T00:01:00Z", "request_count": 120, "bytes": 6000},
            {"timestamp": "2024-01-01T00:02:00Z", "request_count": 130, "bytes": 6500},
            # Add more data points as needed
        ]
    }
    response = client.post("/anomaly-detect", json=test_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "is_anomaly" in json_response
    assert "z_score" in json_response
    assert "reason" in json_response
    assert "current_load" in json_response
    assert "mean_load" in json_response
    assert "std_load" in json_response

if __name__ == "__main__":
    test_health_check()
    test_forecast_endpoint()
    test_anomaly_detection_endpoint()
    print("All tests passed!")
