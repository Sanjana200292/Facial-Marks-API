import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import PIL.Image
from ultralytics import YOLO

app = FastAPI(title="Facial Mark Detection API")

# Enable CORS for React Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both YOLO models into memory
model1 = YOLO("best_model1.pt")
model2 = YOLO("best_model2.pt")


@app.get("/")
def home():
    return {"message": "Facial Mark Detection Dual-Model API is Running on Hugging Face"}


# Endpoint for Model 1
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
        "model_used": "Model 1",
        "total_marks": len(detections),
        "detections": detections,
    }


# Endpoint for Model 2
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
        "model_used": "Model 2",
        "total_marks": len(detections),
        "detections": detections,
    }