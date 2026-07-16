# 🚗 Smart Parking AI

An AI-powered Smart Parking Detection System that identifies **vacant and occupied parking slots** in real time using **YOLOv8**, **OpenCV**, and **Computer Vision**.

## 🌐 Live Demo

🔗 **Streamlit App:** https://smartparkingai-vishakha.streamlit.app

📂 **GitHub Repository:** https://github.com/jhavishakha-tech/SmartParkingAI

---

## 📖 Overview

Finding an empty parking space wastes time, fuel, and increases traffic congestion.

This project uses **a single CCTV camera** to monitor an entire parking area. The system detects vehicles using **YOLOv8**, maps them to predefined parking slots using **OpenCV**, and displays live occupancy information without requiring expensive hardware sensors.

---

## ✨ Features

- 🚗 Real-time Vehicle Detection
- 🟢 Vacant Slot Detection
- 🔴 Occupied Slot Detection
- 📹 Upload Parking Videos
- 📊 Live Parking Statistics
- 💾 Download Processed Video
- 🌐 Streamlit Web Application
- ⚡ Fast YOLOv8 Inference

---

## 🛠 Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- PyTorch

---

## 📂 Project Structure

```
SmartParkingAI
│
├── ai
│   ├── parking_detector.py
│   ├── slot_selector.py
│   └── util.py
│
├── datasets
│   └── busy_parking_lot.mp4
│
├── models
│   └── parking_best.pt
│
├── CarParkPos
├── streamlit_app.py
├── requirements.txt
├── packages.txt
└── README.md
```

---

## ⚙️ Working

1. Upload a parking lot video.
2. YOLOv8 detects all vehicles in every frame.
3. OpenCV compares vehicle locations with predefined parking slot coordinates.
4. Every slot is classified as:
   - 🟢 Vacant
   - 🔴 Occupied
5. The system displays:
   - Total Slots
   - Occupied Slots
   - Vacant Slots
6. Users can download the processed output video.

---

## 📸 Sample Output

- ✅ Green Rectangle → Vacant Slot
- 🔴 Red Rectangle → Occupied Slot
- 📊 Live Parking Count

---

## 🚀 Installation

```bash
git clone https://github.com/jhavishakha-tech/SmartParkingAI.git

cd SmartParkingAI

pip install -r requirements.txt

streamlit run streamlit_app.py
```

---

## 👨‍💻 Future Scope

- Multi-camera parking management
- Automatic Number Plate Recognition (ANPR)
- Mobile App Integration
- Cloud Database
- IoT-based Smart Parking
- Edge AI Deployment

---

## 👩‍💻 Developed By

**Vishakha Jha**

B.Tech – Computer Science & Business Systems

Jain (Deemed-to-be University)

GitHub:
https://github.com/jhavishakha-tech

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!
