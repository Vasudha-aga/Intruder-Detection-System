"""
Integrated Flask Backend with Audio Alerts
Complete working system with sound!
"""

from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import threading
import time
from collections import deque
import os
import numpy as np
import wave

app = Flask(__name__)
CORS(app)

# Global variables
camera = None
detection_active = False
detection_thread = None
alert_playing = False
current_status = {
    'threat_level': 'SAFE',
    'person_count': 0,
    'weapon_count': 0,
    'fps': 0,
    'timestamp': '',
    'alert': False
}

# Create folders
os.makedirs('weapon_alerts', exist_ok=True)

# Generate alert sound if not exists
def generate_alert_sound():
    """Generate siren sound without external dependencies"""
    if os.path.exists('alert.wav'):
        return
    
    print("🔊 Generating alert sound...")
    
    sample_rate = 44100
    duration = 1.5
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create siren effect
    audio = np.array([])
    for i in range(3):
        high_freq = np.sin(2 * np.pi * 1200 * t[:sample_rate//4])
        low_freq = np.sin(2 * np.pi * 600 * t[:sample_rate//4])
        audio = np.concatenate([audio, high_freq, low_freq])
    
    audio = np.int16(audio * 32767 * 0.8)
    
    with wave.open('alert.wav', 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    print("✅ Alert sound generated!")

# Generate sound at startup
generate_alert_sound()

# Load models at startup
print("\n" + "="*60)
print("🔄 Loading AI Models...")
print("="*60)

try:
    person_model = YOLO('yolov8n.pt')
    weapon_model = YOLO('best.pt')
    print("✅ Person Detection Model: Loaded")
    print("✅ Weapon Detection Model: Loaded (96.4% accuracy)")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Make sure yolov8n.pt and best.pt exist!")
    exit()

# Detection histories
person_history = deque(maxlen=5)
weapon_history = deque(maxlen=5)
last_alert_time = 0
last_snapshot_time = 0
ALERT_COOLDOWN = 10

def play_alert_sound(threat_level='SUSPICIOUS'):
    """Play alert sound using pygame (cross-platform)"""
    global alert_playing
    
    if alert_playing:
        return
    
    alert_playing = True
    
    try:
        # Try pygame first (most reliable)
        try:
            import pygame
            pygame.mixer.init()
            sound = pygame.mixer.Sound('alert.wav')
            
            if threat_level == 'CRITICAL':
                # Play twice for critical
                sound.play()
                time.sleep(1.5)
                sound.play()
                time.sleep(1.5)
            else:
                sound.play()
                time.sleep(1.5)
            
        except ImportError:
            # Fallback to playsound
            try:
                from playsound import playsound
                if threat_level == 'CRITICAL':
                    playsound('alert.wav')
                    time.sleep(0.5)
                    playsound('alert.wav')
                else:
                    playsound('alert.wav')
            except:
                # Fallback to system beep
                import winsound
                if threat_level == 'CRITICAL':
                    winsound.Beep(1200, 500)
                    winsound.Beep(600, 500)
                else:
                    winsound.Beep(1000, 500)
                    
    except Exception as e:
        print(f"⚠️ Audio error: {e}")
    
    finally:
        alert_playing = False

def save_snapshot(frame, threat_type, weapon_count):
    """Save image when weapon detected"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"weapon_alerts/{threat_type}_{weapon_count}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")
    return filename

def run_detection():
    """Main detection loop with audio alerts"""
    global camera, detection_active, current_status, last_alert_time, last_snapshot_time
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Could not open camera!")
        detection_active = False
        return
    
    fps_counter = deque(maxlen=30)
    
    print("\n🎥 Camera opened successfully")
    print("🔊 Audio alerts: ENABLED")
    print("🔄 Starting detection loop...")
    
    while detection_active:
        start_time = time.time()
        ret, frame = camera.read()
        
        if not ret:
            break
        
        # Run detections
        person_results = person_model(frame, stream=True, conf=0.5, verbose=False)
        weapon_results = weapon_model(frame, stream=True, conf=0.5, verbose=False)
        
        person_detected = False
        weapon_detected = False
        person_count = 0
        weapon_count = 0
        weapon_types = []
        
        # Person detection
        for r in person_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = person_model.names[cls_id]
                conf = float(box.conf[0])
                
                if label.lower() == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"PERSON {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    person_detected = True
                    person_count += 1
        
        # Weapon detection
        for r in weapon_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = weapon_model.names[cls_id]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"⚠️ {label.upper()} {conf:.2f}",
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                weapon_detected = True
                weapon_count += 1
                weapon_types.append(label)
        
        # Update histories
        person_history.append(person_detected)
        weapon_history.append(weapon_detected)
        
        persistent_person = sum(person_history) >= 3
        persistent_weapon = sum(weapon_history) >= 2
        
        # Threat assessment
        current_time = time.time()
        
        if persistent_weapon:
            threat_level = 'CRITICAL'
            status_color = (0, 0, 255)
            status_text = f"🚨 CRITICAL: {weapon_count} WEAPON(S)"
            
            # Trigger alert
            if not alert_playing and (current_time - last_alert_time > ALERT_COOLDOWN):
                print(f"🚨🚨 CRITICAL ALERT! Weapons: {', '.join(weapon_types)}")
                threading.Thread(target=play_alert_sound, args=('CRITICAL',), daemon=True).start()
                last_alert_time = current_time
                
                # Save snapshot
                if current_time - last_snapshot_time > 5:
                    save_snapshot(frame, 'WEAPON', weapon_count)
                    last_snapshot_time = current_time
                    
        elif persistent_person:
            threat_level = 'SUSPICIOUS'
            status_color = (0, 165, 255)
            status_text = f"⚠️ SUSPICIOUS: {person_count} PERSON(S)"
            
            # Trigger alert
            if not alert_playing and (current_time - last_alert_time > ALERT_COOLDOWN):
                print(f"⚠️ SUSPICIOUS ALERT! Persons: {person_count}")
                threading.Thread(target=play_alert_sound, args=('SUSPICIOUS',), daemon=True).start()
                last_alert_time = current_time
                
        else:
            threat_level = 'SAFE'
            status_color = (0, 255, 0)
            status_text = "✅ AREA SECURE"
        
        # Update global status
        fps = 1 / (time.time() - start_time) if time.time() - start_time > 0 else 0
        fps_counter.append(fps)
        avg_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
        
        current_status.update({
            'threat_level': threat_level,
            'person_count': person_count,
            'weapon_count': weapon_count,
            'fps': round(avg_fps, 1),
            'timestamp': time.strftime('%H:%M:%S'),
            'alert': persistent_weapon or persistent_person
        })
        
        # Draw UI elements
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
        
        cv2.putText(frame, f"THREAT: {threat_level}",
                   (frame.shape[1] - 220, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                   (frame.shape[1] - 150, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'),
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Persons: {person_count} | Weapons: {weapon_count}",
                   (10, frame.shape[0] - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert indicator
        if alert_playing:
            cv2.putText(frame, "🔊 ALERT PLAYING",
                       (frame.shape[1] - 200, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("Army Weapon Detection System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detection_active = False
            break
    
    # Cleanup
    if camera:
        camera.release()
        camera = None
    cv2.destroyAllWindows()
    print("\n📹 Camera closed")

@app.route('/')
def index():
    """Serve the dashboard"""
    return render_template('dashboard.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start detection system"""
    global detection_active, detection_thread
    
    if detection_active:
        return jsonify({
            'status': 'error',
            'message': 'Detection already running'
        })
    
    detection_active = True
    
    # Start detection in separate thread
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    
    print("\n✅ Detection system started from dashboard")
    print("🔊 Audio alerts enabled")
    
    return jsonify({
        'status': 'success',
        'message': 'Detection system started successfully'
    })

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    global detection_active, camera
    
    detection_active = False
    
    # Wait for thread to finish
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=2)
    
    if camera:
        camera.release()
        camera = None
    
    cv2.destroyAllWindows()
    
    print("\n⏹️ Detection system stopped from dashboard")
    
    return jsonify({
        'status': 'success',
        'message': 'Detection system stopped successfully'
    })

@app.route('/status')
def get_status():
    """Get current detection status"""
    return jsonify({
        **current_status,
        'is_active': detection_active
    })

@app.route('/system_info')
def system_info():
    """Get system information"""
    return jsonify({
        'model_accuracy': 96.4,
        'precision': 93.8,
        'recall': 94.3,
        'fps': 28,
        'false_positive_rate': 6.2,
        'dataset_size': 6788,
        'training_epochs': 30,
        'models_loaded': True,
        'audio_enabled': True
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️ WEAPON DETECTION SYSTEM - BACKEND SERVER")
    print("="*60)
    print("\n✅ System ready!")
    print("🔊 Audio alerts: ENABLED")
    print("\n🌐 Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print("\n📋 Instructions:")
    print("   1. Open dashboard in browser")
    print("   2. Click 'START DETECTION' button")
    print("   3. Camera window will open automatically")
    print("   4. Audio alerts will play on detection")
    print("   5. Press 'Q' in camera window to stop")
    print("\n⌨️  Press Ctrl+C to stop server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)