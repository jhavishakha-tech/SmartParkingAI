import os
import time
import pickle
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO


# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "parking_best.pt")
POS_PATH = os.path.join(BASE_DIR, "CarParkPos")

FRAME_W = 1280
FRAME_H = 720


# ==========================================================
# SAME AS LOCAL util.py
# ==========================================================
SLOT_WIDTH = 20
SLOT_HEIGHT = 42
OVERLAP_THRESHOLD = 0.10


def intersection_area(ax1, ay1, ax2, ay2,
                      bx1, by1, bx2, by2):

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)


def check_parking_space(
    img,
    posList,
    car_boxes,
    w=SLOT_WIDTH,
    h=SLOT_HEIGHT,
    overlap_thresh=OVERLAP_THRESHOLD,
):

    free = 0

    slot_area = w * h

    for pos in posList:

        x, y = pos

        sx1 = x
        sy1 = y
        sx2 = x + w
        sy2 = y + h

        occupied = False

        for (cx1, cy1, cx2, cy2) in car_boxes:

            # CAR CENTER
            center_x = (cx1 + cx2) // 2
            center_y = (cy1 + cy2) // 2

            if (
                sx1 <= center_x <= sx2
                and
                sy1 <= center_y <= sy2
            ):
                occupied = True
                break

            inter = intersection_area(
                sx1,
                sy1,
                sx2,
                sy2,
                cx1,
                cy1,
                cx2,
                cy2,
            )

            overlap = inter / slot_area

            if overlap >= overlap_thresh:
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            free += 1

        cv2.rectangle(
            img,
            (sx1, sy1),
            (sx2, sy2),
            color,
            2,
        )

    return free


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_data
def load_slots():
    with open(POS_PATH, "rb") as f:
        return pickle.load(f)
    # ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Smart Parking AI")

model = load_model()
posList = load_slots()

st.success(f"Loaded {len(posList)} parking slots from CarParkPos.")

with st.sidebar:

    st.header("Settings")

    conf = st.slider(
        "Detection confidence",
        0.10,
        0.90,
        0.20,
        0.05,
    )

    frame_skip = st.slider(
        "Process every Nth frame",
        1,
        10,
        2,
    )

    max_frames = st.slider(
        "Max frames to process",
        30,
        1500,
        300,
        30,
    )

uploaded = st.file_uploader(
    "Upload a parking lot video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded is None:

    st.info("Upload a video to begin.")
    st.stop()

temp_input = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)

temp_input.write(uploaded.read())
temp_input.close()

cap = cv2.VideoCapture(temp_input.name)

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output_path = os.path.join(
    tempfile.gettempdir(),
    "output.mp4"
)

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps if fps > 0 else 20,
    (FRAME_W, FRAME_H),
)

col1, col2 = st.columns([3, 1])

frame_window = col1.empty()

free_metric = col2.empty()
occ_metric = col2.empty()
total_metric = col2.empty()

progress = st.progress(0)

run = st.button("▶ Run Detection")
if run:

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_no = 0
    processed = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_no += 1

        if frame_no % frame_skip != 0:
            continue

        if processed >= max_frames:
            break

        frame = cv2.resize(
            frame,
            (FRAME_W, FRAME_H)
        )

        results = model(
            frame,
            imgsz=1280,
            conf=conf,
            verbose=False
        )

        car_boxes = []

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cls = int(box.cls[0])

                # Ignore tiny detections
                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    continue

                car_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                # Vehicle Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

        free = check_parking_space(
            frame,
            posList,
            car_boxes
        )

        total = len(posList)
        occupied = total - free

        cv2.rectangle(
            frame,
            (10, 10),
            (360, 130),
            (40, 40, 40),
            -1
        )

        cv2.putText(
            frame,
            f"Free : {free}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Occupied : {occupied}",
            (20,85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            f"Total : {total}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_window.image(
            rgb,
            channels="RGB",
            use_container_width=True
        )

        free_metric.metric(
            "🟢 Free",
            free
        )

        occ_metric.metric(
            "🔴 Occupied",
            occupied
        )

        total_metric.metric(
            "Total Slots",
            total
        )

        writer.write(frame)

        processed += 1

        progress.progress(
            min(
                processed / max_frames,
                1.0
            )
               )

        cap.release()
        writer.release()

    progress.progress(1.0)

    st.success(
        f"Finished processing {processed} frames."
    )

    with open(output_path, "rb") as f:

        st.download_button(
            label="⬇ Download Processed Video",
            data=f,
            file_name="smart_parking_output.mp4",
            mime="video/mp4",
        )

else:

    st.info(
        "Upload a parking video and click 'Run Detection'."
    )
