import os
import cv2
import pickle
import time
from ultralytics import YOLO
from util import checkParkingSpace, SLOT_WIDTH, SLOT_HEIGHT

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "parking_best.pt")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "datasets", "busy_parking_lot.mp4")
POS_PATH = os.path.join(PROJECT_ROOT, "CarParkPos")

# =====================================================
# LOAD MODEL
# =====================================================
model = YOLO(MODEL_PATH)

# =====================================================
# LOAD PARKING POSITIONS
# =====================================================
if not os.path.exists(POS_PATH):
    print("CarParkPos not found!")
    exit()

with open(POS_PATH, "rb") as f:
    posList = pickle.load(f)

print("Total Slots:", len(posList))

# =====================================================
# OPEN VIDEO
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video!")
    exit()

# Skip black intro
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

prev_time = time.time()

# =====================================================
# MAIN LOOP
# =====================================================
while True:

    success, img = cap.read()

    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        continue

    img = cv2.resize(img, (1280, 720))

    # =================================================
    # YOLO
    # =================================================
    results = model(
        img,
        imgsz=1280,
        conf=0.20,
        verbose=False
    )

    car_boxes = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bw = x2 - x1
            bh = y2 - y1

            # Remove tiny false detections
            if bw < 12 or bh < 12:
                continue

            car_boxes.append((x1, y1, x2, y2))

    # =================================================
    # PARKING CHECK
    # =================================================
    free = checkParkingSpace(
        img,
        posList,
        car_boxes,
        SLOT_WIDTH,
        SLOT_HEIGHT
    )

    total = len(posList)
    occupied = total - free

    # =================================================
    # FPS
    # =================================================
    current = time.time()
    fps = int(1 / max(current - prev_time, 0.001))
    prev_time = current

    # =================================================
    # TOP PANEL
    # =================================================
    cv2.rectangle(
        img,
        (10, 10),
        (340, 120),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        img,
        f"Free : {free}/{total}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Occupied : {occupied}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        img,
        f"FPS : {fps}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =================================================
    # SHOW
    # =================================================
    cv2.imshow("Smart Parking AI", img)

    key = cv2.waitKey(20)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()