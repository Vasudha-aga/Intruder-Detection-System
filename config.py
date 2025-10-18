# Simple Configuration File for Intruder Detection

# Camera Settings
CAMERA_SOURCE = 0  # 0 for webcam, or RTSP URL like 'rtsp://192.168.1.100/stream'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # 0.0 to 1.0 (higher = fewer false alarms)
MODEL_NAME = 'yolov8n.pt'  # yolov8n (fast) or yolov8s (accurate)

# Alert Settings
ALERT_COOLDOWN = 10  # seconds between alerts
ALERT_SOUND = 'alert.mp3'
MIN_DETECTIONS_FOR_ALERT = 3  # Detect in 3 out of 5 frames

# Display Settings
SHOW_FPS = True
SHOW_TIMESTAMP = True
WINDOW_NAME = "Army Intruder Detection System"

# Advanced Settings (Optional)
SAVE_SNAPSHOTS = False  # Save images when intruder detected
SNAPSHOT_FOLDER = 'alerts/'
LOG_FILE = 'detection_log.txt'