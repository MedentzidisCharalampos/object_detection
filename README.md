# YOLO Detection, Segmentation & Pose Estimation Platform
## 🧩 Overview
A modular platform built with **FastAPI** + **Streamlit** to perform:
- Object Detection (YOLOv8)
- Instance Segmentation (YOLOv8-seg)
- Pose Estimation (YOLOv8-pose)
- Video Frame Inference (Detection / Segmentation / Pose)

---


![frontend_layout](frontend_layout.png)
## 🛠️ Setup

```bash
conda create --name vision_models python=3.8
conda activate vision_models

pip install -r requirements.txt
```

### 🔹 Initialize Database
Run once to create the SQLite tables:

```bash
python create_db.py
```

---

## ▶️ Run Services


### 🔹 Start Backend
```bash
uvicorn backend.main:app --reload
```

### 🔹 Start Frontend
```bash
streamlit run frontend/app.py
```

---


## 📥 Pretrained Models
Place these files in a `models/` folder or project root:

- [`yolov8n.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- [`yolov8n-seg.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)
- [`yolov8n-pose.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)


---
## Deployment with Nginx
1. Install Nginx:
```bash
sudo apt update && sudo apt install nginx
```

2. Copy the configuration and generate a self-signed certificate:
```bash
sudo cp nginx/fastapi.conf /etc/nginx/conf.d/
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/nginx/ssl/selfsigned.key -out /etc/nginx/ssl/selfsigned.crt
```
For production, use Let's Encrypt instead:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain
```

3. Start the application servers:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
streamlit run frontend/app.py --server.port 8501
```

4. Reload Nginx:
```bash
sudo nginx -s reload
```

Requests to https://<host>/api will reach FastAPI and / will display the Streamlit UI.


## 📜 License
MIT License. Ultralytics YOLO models used under their respective licenses.
