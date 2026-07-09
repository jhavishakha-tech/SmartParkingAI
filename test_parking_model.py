from ultralytics import YOLO
import cv2

model = YOLO("models/parking_best.pt")

img = cv2.imread("datasets/parking.jpg")

results = model(
    img,
    conf=0.15,
    imgsz=1280
)

print("Results:", len(results))

for r in results:
    print("Boxes:", len(r.boxes))

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        print(cls, conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()