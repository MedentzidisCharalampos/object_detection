# YOLO Segmentation and Detection Platform

## 🧩 Overview
A modular platform built with **FastAPI** + **Streamlit** to perform:
- Object Detection (YOLOv5)
- Instance Segmentation (YOLOv8-seg)
- Image Classification (ResNet)
- Pose Estimation (YOLOv8-pose)
- Video Frame Inference (Detection / Segmentation / Classification / Pose)

---

## 🛠️ Setup

### ✅ Python (manual)
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### ✅ Redis (via Docker)
```bash
docker run -d -p 6379:6379 redis
```

---

## ▶️ Run Services

### 🔹 Run Celery Worker
```bash
celery -A worker.tasks worker --loglevel=info
```

### 🔹 Start Backend
```bash
uvicorn backend.main:app --reload
```

### 🔹 Start Frontend
```bash
streamlit run frontend/app.py
```

---

## 🐳 Docker Compose (Recommended)
```bash
docker-compose up --build
```

---

## 🎯 Endpoints (FastAPI)
Visit [http://localhost:8000/docs](http://localhost:8000/docs)

| Endpoint     | Description              |
|--------------|--------------------------|
| `/detect`    | YOLOv5 Object Detection  |
| `/segment`   | YOLOv8 Instance Segmentation |
| `/classify`  | ResNet Image Classification |
| `/pose`      | YOLOv8 Pose Estimation   |

---

## 📥 Pretrained Models
Place these files in a `models/` folder or project root:

- [`yolov5s.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s.pt)
- [`yolov8n-seg.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)
- [`yolov8n-pose.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)

ResNet-18:
```python
import torch, torchvision.models as models
torch.save(models.resnet18(pretrained=True), "resnet18.pt")
```

---

## 🚀 Features
- FastAPI with Swagger UI
- Async video inference with Celery
- Streamlit frontend UI (optional)
- Modular model support (YOLOv5/YOLOv8/ResNet)

---

## 🧪 Test
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@sample.jpg"
```

---

## 📂 Project Structure
See `yolo_seg_app/` for full code layout.

---

## 📜 License
MIT License. Ultralytics YOLO models used under their respective licenses.