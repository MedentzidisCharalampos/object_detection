def test_api_response():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    with open("data/sample.jpg", "rb") as f:
        response = client.post("/detect", files={"file": ("sample.jpg", f, "image/jpeg")})
    assert response.status_code == 200
