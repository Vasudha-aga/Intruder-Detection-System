from ultralytics import YOLO
import cv2
from playsound import playsound
import threading
from collections import deque
import time

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Using yolov8n for better speed

# Open webcam
cap = cv2.VideoCapture(0)
alert_playing = False

# Tracking improvements
detection_history = deque(maxlen=5)  # Track last 5 frames
last_alert_time = 0
ALERT_COOLDOWN = 10  # seconds between alerts

def play_alert():
    global alert_playing
    alert_playing = True
    playsound('alert.mp3')
    alert_playing = False

# FPS counter
fps_counter = deque(maxlen=30)
frame_count = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True, conf=0.5, verbose=False)  # Added conf threshold
    intruder_detected = False
    person_count = 0

    # Parse results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Only detect persons with good confidence
            if label.lower() == 'person' and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw red box for intruder
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"INTRUDER {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                intruder_detected = True
                person_count += 1

    # Persistent detection - only alert if detected in multiple frames
    detection_history.append(intruder_detected)
    persistent_detection = sum(detection_history) >= 3  # Detected in 3 out of 5 frames

    # Play alarm with cooldown
    current_time = time.time()
    if persistent_detection and not alert_playing:
        if current_time - last_alert_time > ALERT_COOLDOWN:
            threading.Thread(target=play_alert).start()
            last_alert_time = current_time
            print(f"🚨 ALERT! {person_count} intruder(s) detected at {time.strftime('%H:%M:%S')}")

    # FPS calculation
    fps = 1 / (time.time() - start_time)
    fps_counter.append(fps)
    avg_fps = sum(fps_counter) / len(fps_counter)

    # Status bar
    status_color = (0, 0, 255) if intruder_detected else (0, 255, 0)
    status_text = f"ALERT: {person_count} INTRUDERS" if intruder_detected else "SECURE"
    
    # Draw status bar at top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    # FPS counter bottom right
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1] - 150, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show timestamp
    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Army Intruder Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped successfully!")