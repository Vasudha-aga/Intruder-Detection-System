from ultralytics import YOLO
import cv2
from playsound import playsound
import threading
from collections import deque
import time
import os

# Create folders for saving alerts
os.makedirs('weapon_alerts', exist_ok=True)

# Load TWO models instead of one
print("🔄 Loading models...")
person_model = YOLO('yolov8n.pt')  # Original person detection
weapon_model = YOLO('best.pt')     # NEW: Weapon detection model
print("✅ Models loaded!")

# Open webcam
cap = cv2.VideoCapture(0)
alert_playing = False

# Tracking improvements - NOW TWO HISTORIES
person_history = deque(maxlen=5)  # Track person detections
weapon_history = deque(maxlen=5)  # NEW: Track weapon detections
last_alert_time = 0
ALERT_COOLDOWN = 10  # seconds between alerts

# NEW: Alert levels with colors
THREAT_COLORS = {
    'SAFE': (0, 255, 0),        # Green
    'SUSPICIOUS': (0, 165, 255),  # Orange
    'CRITICAL': (0, 0, 255)      # Red
}

def play_alert(threat_level='SUSPICIOUS'):
    """Play alert sound based on threat level"""
    global alert_playing
    alert_playing = True
    
    if threat_level == 'CRITICAL':
        # Play twice for critical threats
        for _ in range(2):
            playsound('alert.wav')
    else:
        playsound('alert.wav')
    
    alert_playing = False

def save_snapshot(frame, threat_type, weapon_count):
    """NEW: Save image when weapon detected"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"weapon_alerts/{threat_type}_{weapon_count}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")
    return filename

# FPS counter
fps_counter = deque(maxlen=30)
frame_count = 0

# NEW: Statistics tracking
total_weapon_alerts = 0
last_snapshot_time = 0

print("\n🚀 System Started!")
print("👤 Person Detection: Active")
print("🔫 Weapon Detection: Active")
print("Press 'Q' to quit\n")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run BOTH detections
    person_results = person_model(frame, stream=True, conf=0.5, verbose=False)
    weapon_results = weapon_model(frame, stream=True, conf=0.5, verbose=False)  # NEW
    
    person_detected = False
    weapon_detected = False
    person_count = 0
    weapon_count = 0
    weapon_types = []  # NEW: Store weapon names

    # ===== PERSON DETECTION (Original code) =====
    for r in person_results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = person_model.names[cls_id]
            conf = float(box.conf[0])

            if label.lower() == 'person' and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Yellow box for person (changed from red)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"PERSON {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                person_detected = True
                person_count += 1

    # ===== NEW: WEAPON DETECTION =====
    for r in weapon_results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = weapon_model.names[cls_id]
            conf = float(box.conf[0])

            if conf > 0.5:  # Weapon detected
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # RED box for weapons
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"⚠️ {label.upper()} {conf:.2f}", 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                weapon_detected = True
                weapon_count += 1
                weapon_types.append(label)

    # Persistent detection - NOW FOR BOTH
    person_history.append(person_detected)
    weapon_history.append(weapon_detected)  # NEW
    
    persistent_person = sum(person_history) >= 3
    persistent_weapon = sum(weapon_history) >= 2  # NEW: More strict for weapons

    # NEW: Threat Assessment Logic
    if persistent_weapon:
        threat_level = 'CRITICAL'
        status_color = THREAT_COLORS['CRITICAL']
    elif persistent_person:
        threat_level = 'SUSPICIOUS'
        status_color = THREAT_COLORS['SUSPICIOUS']
    else:
        threat_level = 'SAFE'
        status_color = THREAT_COLORS['SAFE']

    # Play alarm with cooldown
    current_time = time.time()
    if not alert_playing and (current_time - last_alert_time > ALERT_COOLDOWN):
        
        # NEW: Critical alert for weapons
        if persistent_weapon:
            total_weapon_alerts += 1
            threading.Thread(target=play_alert, args=('CRITICAL',)).start()
            last_alert_time = current_time
            
            # Save snapshot (max once every 5 seconds)
            if current_time - last_snapshot_time > 5:
                save_snapshot(frame, 'WEAPON', weapon_count)
                last_snapshot_time = current_time
            
            print(f"🚨🚨 CRITICAL! {weapon_count} WEAPON(S): {', '.join(weapon_types)} at {time.strftime('%H:%M:%S')}")
        
        # Person only alert
        elif persistent_person:
            threading.Thread(target=play_alert, args=('SUSPICIOUS',)).start()
            last_alert_time = current_time
            print(f"⚠️ ALERT! {person_count} intruder(s) detected at {time.strftime('%H:%M:%S')}")

    # FPS calculation
    fps = 1 / (time.time() - start_time)
    fps_counter.append(fps)
    avg_fps = sum(fps_counter) / len(fps_counter)

    # NEW: Enhanced status bar
    if threat_level == 'CRITICAL':
        status_text = f"🚨 CRITICAL: {weapon_count} WEAPON(S) + {person_count} PERSON(S)"
    elif threat_level == 'SUSPICIOUS':
        status_text = f"⚠️ SUSPICIOUS: {person_count} PERSON(S) DETECTED"
    else:
        status_text = "✅ AREA SECURE"
    
    # Draw status bar at top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
    
    # NEW: Threat level indicator (top right)
    cv2.putText(frame, f"THREAT: {threat_level}", 
                (frame.shape[1] - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # FPS counter bottom right
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1] - 150, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show timestamp
    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # NEW: Detection counters
    cv2.putText(frame, f"Persons: {person_count} | Weapons: {weapon_count}", 
                (10, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Army Intruder + Weapon Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ System stopped successfully!")
print(f"📊 Total weapon alerts: {total_weapon_alerts}")
print(f"📁 Saved images in: weapon_alerts/")