import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import PIL.Image
from ultralytics import YOLO

app = FastAPI(title="Facial Mark Detection API")

# Enable cors for all access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO("best.pt")


@app.get("/")
def home():
    return {"message": "Facial Mark Detection API is Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))

    # Prediction execution
    results = model.predict(image, conf=0.25)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf) * 100, 2),
            "bbox": box.xywh.tolist()[0],
        })

    return {
        "status": "success",
        "total_marks": len(detections),
        "detections": detections,
    }