from fastapi.testclient import TestClient
import io
from PIL import Image

from backend.main import app

client = TestClient(app)

def test_detect_endpoint(monkeypatch):
    def fake_handle_image(image_bytes, task):
        assert task == 'detect'
        return {'result_path': 'fake.jpg', 'class_names': ['person']}

    monkeypatch.setattr('backend.main.handle_image', fake_handle_image)

    img = Image.new('RGB', (1, 1), color='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    files = {'file': ('test.jpg', buf, 'image/jpeg')}
    response = client.post('/detect', files=files)
    assert response.status_code == 200
    assert response.json() == {'result_path': 'fake.jpg', 'class_names': ['person']}
