from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models.detector import Detector
from backend.models.segmentor import Segmentor
from backend.models.pose_estimator import PoseEstimator
from backend.service.detection_service import handle_image
from backend.service.video_service import handle_video
from starlette.responses import JSONResponse

app = FastAPI(title="YOLO Multi-Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector("models/yolov8n.pt")
segmentor = Segmentor("models/yolov8n-seg.pt")
pose_model = PoseEstimator("models/yolov8n-pose.pt")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = handle_image(image_bytes, task="detect")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = handle_image(image_bytes, task="segment")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pose")
async def pose_estimate(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = handle_image(image_bytes, task="pose")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/detect")
async def video_detect(file: UploadFile = File(...)):
    try:
        result = await handle_video(file, "detect", detector=detector)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/segment")
async def video_segment(file: UploadFile = File(...)):
    try:
        result = await handle_video(file, "segment", segmentor=segmentor)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/pose")
async def video_pose(file: UploadFile = File(...)):
    try:
        result = await handle_video(file, "pose", pose_estimator=pose_model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
