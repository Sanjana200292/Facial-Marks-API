import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import PIL.Image
from ultralytics import YOLO

app = FastAPI(title="Facial Mark Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Direct Links
# ⚠️ වැදගත්: 'your-hf-username' වෙනුවට ඔයාගේ Hugging Face Username එක යොදන්න!
HF_USERNAME = "your-hf-username" 
REPO_NAME = "facial-mark-detection-models"

MODEL1_URL = f"https://huggingface.co/{HF_USERNAME}/{REPO_NAME}/resolve/main/best_model1.pt"
MODEL2_URL = f"https://huggingface.co/{HF_USERNAME}/{REPO_NAME}/resolve/main/best_model2.pt"

# Server එක Run වෙද්දී Hugging Face එකෙන් Models auto-load වේ
model1 = YOLO(MODEL1_URL)  # YOLOv8
model2 = YOLO(MODEL2_URL)  # YOLOv11


@app.get("/")
def home():
    return {"message": "Facial Mark Detection Dual-Model API is Running"}


# Endpoint for Model 1 (YOLOv8)
@app.post("/predict/model1")
async def predict_model1(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))

    results = model1.predict(image, conf=0.25)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model1.names[int(box.cls)],
            "confidence": round(float(box.conf) * 100, 2),
            "bbox": box.xywh.tolist()[0],
        })

    return {
        "status": "success",
        "model_used": "YOLOv8",
        "total_marks": len(detections),
        "detections": detections,
    }


# Endpoint for Model 2 (YOLOv11)
@app.post("/predict/model2")
async def predict_model2(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))

    results = model2.predict(image, conf=0.25)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model2.names[int(box.cls)],
            "confidence": round(float(box.conf) * 100, 2),
            "bbox": box.xywh.tolist()[0],
        })

    return {
        "status": "success",
        "model_used": "YOLOv11",
        "total_marks": len(detections),
        "detections": detections,
    }