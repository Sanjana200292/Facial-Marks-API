import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import PIL.Image
from ultralytics import YOLO

app = FastAPI(title="Facial Mark Detection Dual-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models load කිරීම (root folder එකේ තියෙන නිසා direct filenames භාවිත කර ඇත)
model1 = YOLO("best_model1.pt")  # YOLOv8
model2 = YOLO("best_model2.pt")  # YOLOv11


@app.get("/")
def home():
    return {"message": "Facial Mark Detection Dual-Model API is Running"}


# Helper function: Model එකකින් Detection Results ලබාගන්නා logic එක
def run_prediction(model, image):
    results = model.predict(image, conf=0.25)
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf) * 100, 2),
            "bbox": box.xywh.tolist()[0],
        })
    return detections


# 1. Combined Endpoint (එකම Request එකෙන් Models 2කේම Output එක ගන්න)
@app.post("/predict/both")
async def predict_both(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))

    yolov8_results = run_prediction(model1, image)
    yolov11_results = run_prediction(model2, image)

    return {
        "status": "success",
        "models": {
            "yolov8": {
                "total_marks": len(yolov8_results),
                "detections": yolov8_results,
            },
            "yolov11": {
                "total_marks": len(yolov11_results),
                "detections": yolov11_results,
            },
        },
    }


# 2. Endpoint for Model 1 only (YOLOv8)
@app.post("/predict/model1")
async def predict_model1(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))
    yolov8_results = run_prediction(model1, image)

    return {
        "status": "success",
        "model_used": "YOLOv8",
        "total_marks": len(yolov8_results),
        "detections": yolov8_results,
    }


# 3. Endpoint for Model 2 only (YOLOv11)
@app.post("/predict/model2")
async def predict_model2(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))
    yolov11_results = run_prediction(model2, image)

    return {
        "status": "success",
        "model_used": "YOLOv11",
        "total_marks": len(yolov11_results),
        "detections": yolov11_results,
    }