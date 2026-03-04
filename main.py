from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import torch
import uvicorn

app = FastAPI(title="YOLO CPU Detection API")

# Force CPU
device = "cpu"

# Load model once at startup
model = YOLO("models/best_box_14022026.pt")
model.to(device)

@app.get("/")
def health():
    return {"status": "running", "device": device}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(image, device=device)

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return {"detections": detections}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

# 👇 THIS PART is required for python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)