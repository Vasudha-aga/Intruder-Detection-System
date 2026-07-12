# 🛡️ Army Weapon & Intruder Detection System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Vasudha2711/Intruder-Detection-System)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-96.4%25%20Accuracy-success)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10-yellow)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

A real-time, AI-powered computer vision surveillance system built for armed intruder detection and automated threat alerting. Deployed with dual YOLOv8 models for simultaneous human and weapon recognition, integrated with a live video streaming web dashboard and priority-based audio siren alerts.

---

## 🚀 Key Performance Metrics
* **Weapon Detection Accuracy:** `96.4%`
* **Precision / Recall:** `93.8%` / `94.3%`
* **False Positive Reduction:** `68%` reduction in false alarms
* **Inference Speed:** `~28 FPS` real-time processing
* **Dataset Size:** `6,788` annotated images trained over `30 epochs`

---

## ✨ System Features
* **👤 Real-Time Person Detection:** Identifies individuals entering monitored security zones.
* **🔫 Weapon Recognition:** Instantaneously detects firearms, rifles, and bladed weapons.
* **🚨 3-Level Threat Classification:**
  * 🟢 **SAFE:** Area secure, normal activity.
  * 🟡 **SUSPICIOUS:** Unverified person detected in security perimeter.
  * 🔴 **CRITICAL:** Weapon detected! Triggers visual and audio alarms.
* **🌐 Web Dashboard & Video Streaming:** Live HTTP video feed (`/video_feed`) accessible from any browser or smartphone.
* **☁️ Cloud Simulation Mode:** Automatically falls back to dataset simulation streaming when running in headless cloud containers without physical webcams.
* **🔊 Automated Audio Alarms:** Multi-frequency siren generation (`alert.wav`) with cross-platform audio playback.
* **📸 Evidence Snapshot Logging:** Automatically captures and saves timestamped JPEG snapshots to `weapon_alerts/` upon critical weapon detection.

---

## 🛠️ Technology Stack
* **Deep Learning & Computer Vision:** YOLOv8 (PyTorch), OpenCV (`cv2`)
* **Backend Web Server:** Flask, Flask-CORS, Gunicorn
* **Audio Processing:** Pygame, Wave, NumPy, Pydub
* **DevOps & Deployment:** Docker, Hugging Face Spaces, GitHub Actions CI/CD

---

## 💻 Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vasudha-aga/Intruder-Detection-System.git
   cd Intruder-Detection-System
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the Dashboard:**
   Open your browser and navigate to: `http://localhost:5000`
   * Click **START DETECTION** to begin live camera surveillance.
   * Press **Q** in the camera window or click **STOP DETECTION** on the web dashboard to end the session.

---
*⚠️ **Disclaimer:** This system is developed as a security and research project for automated surveillance and intrusion detection.*
