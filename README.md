# 🚗 Smart Parking AI

An AI-powered Smart Parking Detection System that detects **vacant and occupied parking slots** in real time using **YOLOv8**, **OpenCV**, and **Streamlit**.

---

## 🌐 Live Demo

🚀 **Live Application:**  
https://smartparkingai-vishakha.streamlit.app

📂 **GitHub Repository:**  
https://github.com/jhavishakha-tech/SmartParkingAI

---

## 📖 Project Overview

Finding an empty parking space in shopping malls, airports, colleges, and office buildings is time-consuming and often leads to fuel wastage and traffic congestion.

This project uses **Computer Vision** and **Artificial Intelligence** to automatically detect available parking spaces from a CCTV camera feed.

Instead of installing expensive parking sensors in every slot, the system analyzes the video feed using **YOLOv8** and **OpenCV** to identify whether each parking slot is **Vacant** or **Occupied** in real time.

---

## ✨ Features

- 🚗 Real-Time Vehicle Detection
- 🟢 Vacant Parking Slot Detection
- 🔴 Occupied Parking Slot Detection
- 📊 Live Parking Statistics
- 🎥 Upload Parking Videos
- 📥 Download Processed Output Video
- ⚡ YOLOv8 Based Object Detection
- 🌐 Streamlit Web Interface
- 📈 Real-Time Visualization

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- PyTorch
- Computer Vision

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
│   ├── busy_parking_lot.mp4
│   └── parking.jpg
│
├── models
│   ├── parking_best.pt
│   └── yolov8s.pt
│
├── CarParkPos
├── streamlit_app.py
├── requirements.txt
├── packages.txt
├── extract_frame.py
└── README.md
```

---

# ⚙️ Working

### Step 1

The user uploads a parking lot video through the Streamlit web application.

### Step 2

The uploaded video is processed frame by frame.

### Step 3

YOLOv8 detects all vehicles present in every frame.

### Step 4

OpenCV compares detected vehicle positions with predefined parking slot coordinates stored inside **CarParkPos**.

### Step 5

Each parking slot is classified as:

🟢 Vacant

or

🔴 Occupied

### Step 6

The application displays

- Total Parking Slots
- Vacant Slots
- Occupied Slots
- Live Annotated Video

### Step 7

The processed video can be downloaded directly from the browser.

---

## 💡 Advantages

- No extra parking sensors required
- Uses existing CCTV cameras
- Low deployment cost
- Real-Time Monitoring
- Easy to use
- Scalable for large parking lots

---

## 📸 Sample Output

The application displays:

- 🟢 Green Boxes → Vacant Slots
- 🔴 Red Boxes → Occupied Slots
- 📊 Live Parking Count
- 🎥 Processed Video Output

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/jhavishakha-tech/SmartParkingAI.git
```

Go to project directory

```bash
cd SmartParkingAI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run streamlit_app.py
```

---

## 🎯 Future Scope

- Multi-Camera Parking Monitoring
- Automatic Number Plate Recognition (ANPR)
- Mobile Application
- Cloud Database Integration
- IoT Enabled Smart Parking
- Edge AI Deployment
- Live CCTV Streaming Support
- Smart City Integration

---

## 👩‍💻 Developer

**Vishakha Jha**

B.Tech – Computer Science & Business Systems

Jain (Deemed-to-be University)

GitHub Profile:

https://github.com/jhavishakha-tech

---

## 🙏 Acknowledgements

- Ultralytics YOLOv8
- OpenCV
- Streamlit
- Python Community

---

## ⭐ Support

If you found this project useful,

please ⭐ **Star this repository** and share it with others.

Thank you for visiting the project! 🚀
