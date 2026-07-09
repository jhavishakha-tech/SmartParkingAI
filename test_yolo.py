from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8s.pt")

img = cv2.imread("datasets/parking.jpg")

print("Image Shape:", img.shape)

results = model(
    img,
    conf=0.10,
    imgsz=1280,
    verbose=False
)

print("Results:", len(results))

for r in results:
    print("Boxes:", len(r.boxes))

    for box in r.boxes:
        print(
            int(box.cls[0]),
            float(box.conf[0])
        )