import os
import cv2
import pickle
from util import SLOT_WIDTH, SLOT_HEIGHT

# ---------------------------------
# Paths
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

VIDEO_PATH = os.path.join(PROJECT_ROOT, "datasets", "busy_parking_lot.mp4")
POS_PATH = os.path.join(PROJECT_ROOT, "CarParkPos")

# ---------------------------------
# Load Existing Slots
# ---------------------------------
if os.path.exists(POS_PATH):
    with open(POS_PATH, "rb") as f:
        posList = pickle.load(f)
else:
    posList = []

# ---------------------------------
# Read Video Frame
# ---------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video.")
    exit()

# Skip black intro frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

ret, img = cap.read()
cap.release()

if not ret:
    print("Failed to read video frame.")
    exit()

img = cv2.resize(img, (1280, 720))

# ---------------------------------
# Mouse Events
# ---------------------------------
def mouseClick(event, x, y, flags, params):
    global posList

    # Add Slot
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))

        with open(POS_PATH, "wb") as f:
            pickle.dump(posList, f)

    # Delete Slot
    elif event == cv2.EVENT_RBUTTONDOWN:

        for i, pos in enumerate(posList):

            px, py = pos

            if (
                px <= x <= px + SLOT_WIDTH
                and
                py <= y <= py + SLOT_HEIGHT
            ):
                posList.pop(i)

                with open(POS_PATH, "wb") as f:
                    pickle.dump(posList, f)

                break

# ---------------------------------
# Main Loop
# ---------------------------------
while True:

    imgCopy = img.copy()

    for x, y in posList:

        cv2.rectangle(
            imgCopy,
            (x, y),
            (x + SLOT_WIDTH, y + SLOT_HEIGHT),
            (255, 0, 255),
            2,
        )

        cv2.circle(
            imgCopy,
            (x, y),
            3,
            (0, 255, 255),
            -1,
        )

    cv2.putText(
        imgCopy,
        f"Slots : {len(posList)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Parking Slot Selector", imgCopy)

    cv2.setMouseCallback(
        "Parking Slot Selector",
        mouseClick,
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()