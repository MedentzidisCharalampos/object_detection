import os
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "..", "sample_images", "1.jpg")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def _post_image(endpoint: str):
    with open(SAMPLE_IMAGE, "rb") as f:
        return client.post(endpoint, files={"file": ("1.jpg", f, "image/jpeg")})

def test_detect_endpoint():
    response = _post_image("/detect")
    assert response.status_code == 200
    data = response.json()
    assert "result_path" in data
    assert "class_names" in data

def test_segment_endpoint():
    response = _post_image("/segment")
    assert response.status_code == 200
    data = response.json()
    assert "result_path" in data
    assert "class_names" in data

def test_pose_endpoint():
    response = _post_image("/pose")
    assert response.status_code == 200
    data = response.json()
    assert "result_path" in data
    assert "class_names" in data

