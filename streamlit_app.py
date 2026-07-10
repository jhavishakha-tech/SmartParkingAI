import os
import time
import pickle
import tempfile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ============================================================
# Paths (matched to the actual repo layout:
#   /models/parking_best.pt
#   /CarParkPos
#   /ai/util.py  (slot-checking logic, reused if importable)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "parking_best.pt")
POS_PATH = os.path.join(BASE_DIR, "CarParkPos")

FRAME_W = 1280
FRAME_H = 720

SLOT_WIDTH = 38
SLOT_HEIGHT = 78
OVERLAP_THRESHOLD = 0.25
DEFAULT_GRID_ROWS = 5
DEFAULT_GRID_COLS = 8


# ============================================================
# Slot-occupancy logic (same rules as ai/util.py, kept inline
# so this app has no fragile import-path dependency on the ai/
# package when deployed on Streamlit Community Cloud)
# ============================================================
def intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)


def check_parking_space(img, pos_list, car_boxes, w=SLOT_WIDTH, h=SLOT_HEIGHT,
                         overlap_thresh=OVERLAP_THRESHOLD):
    free = 0
    slot_area = w * h

    for (x, y) in pos_list:
        sx1, sy1, sx2, sy2 = x, y, x + w, y + h
        scx, scy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
        occupied = False

        for (cx1, cy1, cx2, cy2) in car_boxes:
            if cx1 <= scx <= cx2 and cy1 <= scy <= cy2:
                occupied = True
                break
            inter = intersection_area(sx1, sy1, sx2, sy2, cx1, cy1, cx2, cy2)
            if inter / slot_area >= overlap_thresh:
                occupied = True
                break

        color = (0, 0, 255) if occupied else (0, 255, 0)
        if not occupied:
            free += 1
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, 2)

    return free


def generate_default_grid(frame_w, frame_h, rows=DEFAULT_GRID_ROWS, cols=DEFAULT_GRID_COLS,
                           slot_w=SLOT_WIDTH, slot_h=SLOT_HEIGHT,
                           margin_x=40, margin_y=40, spacing_x=12, spacing_y=20):
    pos_list = []
    usable_w = frame_w - 2 * margin_x
    usable_h = frame_h - 2 * margin_y
    cell_w = usable_w / cols
    cell_h = usable_h / rows
    for r in range(rows):
        for c in range(cols):
            x = int(margin_x + c * cell_w + (cell_w - slot_w) / 2)
            y = int(margin_y + r * cell_h + (cell_h - slot_h) / 2)
            x = max(0, min(x, frame_w - slot_w))
            y = max(0, min(y, frame_h - slot_h))
            pos_list.append((x, y))
    return pos_list


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. "
                 f"Make sure models/parking_best.pt is committed to the repo "
                 f"(check it isn't excluded by .gitignore or too large for a "
                 f"normal git push -- see Git LFS note below).")
        st.stop()
    return YOLO(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_pos_list():
    """
    Self-healing loader: falls back to an auto-generated grid instead
    of crashing if CarParkPos is missing, empty, or corrupted -- this
    matters especially in a hosted environment where nobody can click
    through a local slot_selector.py to fix it by hand.
    """
    if os.path.exists(POS_PATH):
        try:
            with open(POS_PATH, "rb") as f:
                pos_list = pickle.load(f)
            if isinstance(pos_list, list) and len(pos_list) > 0:
                return pos_list, "CarParkPos"
        except Exception:
            pass

    pos_list = generate_default_grid(FRAME_W, FRAME_H)
    return pos_list, "auto-generated (CarParkPos missing/empty/corrupted)"


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Smart Parking AI", page_icon="🚗", layout="wide")

st.title("🚗 Smart Parking AI")
st.caption("YOLOv8 + OpenCV parking slot occupancy detector — Streamlit edition")

with st.sidebar:
    st.header("Settings")
    conf_thresh = st.slider("Detection confidence", 0.05, 0.90, 0.20, 0.05)
    frame_stride = st.slider(
        "Process every Nth frame", 1, 10, 2,
        help="Higher = faster processing, lower resolution in time. "
             "Community Cloud is CPU-only, so skipping frames keeps this responsive."
    )
    max_frames = st.slider(
        "Max frames to process", 30, 1500, 300, step=30,
        help="Caps total runtime so the app doesn't time out on long videos."
    )
    st.markdown("---")
    st.caption("Model: `models/parking_best.pt`")
    st.caption("Slots: `CarParkPos`")

model = load_model()
pos_list, pos_source = load_pos_list()

if "auto-generated" in pos_source:
    st.warning(
        f"⚠️ Using an auto-generated slot grid ({len(pos_list)} slots) because "
        f"CarParkPos was missing or empty. Occupancy positions will be approximate. "
        f"Commit a valid CarParkPos file to the repo for exact slot alignment."
    )
else:
    st.success(f"Loaded {len(pos_list)} parking slots from CarParkPos.")

uploaded_video = st.file_uploader(
    "Upload a parking lot video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:

    # Persist the upload to a temp file so cv2.VideoCapture can read it
    in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_tmp.write(uploaded_video.read())
    in_tmp.close()
    input_path = in_tmp.name

    out_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video. Try a different file/format.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frames_to_process = min(total_frames, max_frames * frame_stride)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = max(1, src_fps / frame_stride)
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (FRAME_W, FRAME_H))

    col1, col2 = st.columns([3, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        free_metric = st.empty()
        occ_metric = st.empty()
        total_metric = st.empty()

    progress_bar = st.progress(0, text="Starting...")
    start_btn_area = st.empty()

    run = start_btn_area.button("▶ Run detection", type="primary")

    if run:
        processed_count = 0
        frame_idx = 0
        prev_time = time.time()

        while frame_idx < frames_to_process:
            success, frame = cap.read()
            if not success or frame is None:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            results = model(frame, imgsz=1280, conf=conf_thresh, iou=0.45, verbose=False)

            car_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) < 15 or (y2 - y1) < 15:
                        continue
                    car_boxes.append((x1, y1, x2, y2))

            free = check_parking_space(frame, pos_list, car_boxes)
            total = len(pos_list)
            occupied = total - free

            cv2.rectangle(frame, (10, 10), (340, 130), (40, 40, 40), -1)
            cv2.putText(frame, f"Free : {free}/{total}", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Occupied : {occupied}", (25, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            now = time.time()
            fps = 1 / max(now - prev_time, 0.001)
            prev_time = now
            cv2.putText(frame, f"FPS : {fps:.1f}", (25, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            free_metric.metric("🟢 Free", free)
            occ_metric.metric("🔴 Occupied", occupied)
            total_metric.metric("Total Slots", total)

            processed_count += 1
            frame_idx += 1
            progress_bar.progress(
                min(frame_idx / frames_to_process, 1.0),
                text=f"Processed frame {frame_idx}/{frames_to_process}"
            )

        cap.release()
        writer.release()
        progress_bar.progress(1.0, text="Done.")

        st.success(f"Finished processing {processed_count} frames.")

        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇ Download processed video",
                data=f,
                file_name="smart_parking_output.mp4",
                mime="video/mp4",
            )
else:
    st.info("Upload a video above, then click **Run detection** to start.")
