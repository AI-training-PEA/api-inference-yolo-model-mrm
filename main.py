from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import torch
import uvicorn
import cv2
import numpy as np

num_map = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}


app = FastAPI(title="YOLO CPU Detection API")

# Force CPU
device = "cpu"

# Load model once at startup
model_box = YOLO("models/best_box_14022026.pt")
model_box.to(device)

model_numbers = YOLO("models/best_numbers_trainlocal_01032026.pt")
model_numbers.to(device)

# =========================
# Helper Functions
# =========================
def line_intersection(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px,py])


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect




def perspective_warp(original, results=None, display="N"):

    # try:
    # =========================
    # 1. Default ROI
    # =========================
    if original is None or original.size == 0:
        return original, original
    # Convert PIL to numpy array
    roi = np.array(original)

    # roi = original.copy()

    # =========================
    # 2. YOLO Crop (Safe Mode)
    # =========================
    try:
        if results is not None and hasattr(results, "boxes"):
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                conf = boxes.conf.cpu().numpy()
                best_idx = np.argmax(conf)

                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

                # Clamp boundary
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original.shape[1], x2)
                y2 = min(original.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    roi = original[y1:y2, x1:x2]

    except Exception:
        roi = original.copy()

    if roi is None or roi.size == 0:
        return original, original

    # =========================
    # 3. Edge Detection
    # =========================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # =========================
    # 4. Hough Lines
    # =========================
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180, 
        threshold=100,
        minLineLength=100,
        maxLineGap=20
    )

    if lines is None:
        return roi, roi

    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2((y2-y1), (x2-x1)))

        if abs(angle) < 20:
            horizontal_lines.append(line[0])
        elif abs(angle) > 70:
            vertical_lines.append(line[0])

    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        return roi, roi

    # =========================
    # 5. Line Intersection
    # =========================
    points = []

    for v in vertical_lines:
        for h in horizontal_lines:
            pt = line_intersection(v,h)
            if pt is not None:
                if 0 <= pt[0] <= roi.shape[1] and 0 <= pt[1] <= roi.shape[0]:
                    points.append(pt)

    if len(points) < 4:
        return roi, roi

    points = np.array(points, dtype=np.float32)

    # =========================
    # 6. Select Quad (Preserved Logic)
    # =========================
    hull = cv2.convexHull(points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) != 4:
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        quad = np.array(box, dtype="float32")
    else:
        quad = approx.reshape(-1,2).astype("float32")

    rect_pts = order_points(quad)

    # =========================
    # 7. Warp (Preserved)
    # =========================
    (tl,tr,br,bl) = rect_pts

    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA,widthB))

    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA,heightB))

    if maxWidth < 10 or maxHeight < 10:
        return roi, roi

    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]
    ], dtype="float32")

    H = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(roi, H, (maxWidth,maxHeight))

    if warped is None or warped.size == 0:
        return roi, roi


    return roi, warped
    # except Exception as e:
    #     print("Perspective_Meter crashed → fallback")
    #     print("Error:", e)
    #     return original, original
    
    


@app.get("/")
def health():
    return {"status": "running", "device": device}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
        _, perspective_image = perspective_warp(raw_image)
        image = Image.fromarray(perspective_image)

        results = model_box(image, device=device)
        results = model_box.predict(image, conf=0.25)
        result = results[0]
        print(len(result.boxes))

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            # พิกัด (left,top,right,bottom) หรือ (x1, y1, x2, y2) ในภาพ
            cords = (x1, y1, x2, y2)

        # ครอปภาพจาก model ตัวที่ 1 (ตรวจจับกรอบ)เพื่อส่งให้ model ตัวที่ 2 (อ่านหน่วย)
        # img = cv2.imread(image_path)
        #crop_img = image[y1:y2, x1:x2]
        crop_img = image.crop((x1, y1, x2, y2))
        # model kwhnumbers predict
        results = model_numbers.predict(crop_img, conf=0.20)
        result = results[0]
        print(len(result.boxes))
        digits = []  # เก็บ (x1, digit)
        # เปลี่ยนชื่อตัวแปร
        img = crop_img

        for i, box in enumerate(result.boxes):
            class_name = model_numbers.names[int(box.cls[0])]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            print(f"Box: {i+1}")

            # แสดงข้อมูลแต่ละกล่อง
            cords = (x1, y1, x2, y2)
            print("Class Name:", class_name)
            print("Coordinates:", cords)
            print("Confidence:", round(conf, 2))
            print("---")


            # ADD เก็บ digit หลัง mapping
            if class_name in num_map:
                digits.append((x1, num_map[class_name]))

        # รวมตัวเลขเรียงตามตำแหน่ง x1 แล้ว print ออกมา
        digits_str = ''.join(d for _, d in sorted(digits, key=lambda t: t[0]))
        print(f"\nkWh unit: {digits_str}\n")

        return {"detections": digits_str}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

# 👇 THIS PART is required for python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)