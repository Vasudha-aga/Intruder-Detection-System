"""
Fixed Weapon Detection Training Script
Corrects path issues for data/ folder structure
"""

from ultralytics import YOLO
import os
import shutil

print("="*60)
print("🔫 WEAPON DETECTION MODEL TRAINING - FIXED PATHS")
print("="*60)

# Check current directory
current_dir = os.getcwd()
print(f"\n📍 Working directory: {current_dir}")

# Check if data.yaml exists
if not os.path.exists('data.yaml'):
    print("\n⚠️  data.yaml not found! Creating it...")
    
    yaml_content = """# YOLOv8 Weapon Detection Dataset Configuration

path: data
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['weapon']
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✅ data.yaml created with correct paths")

# Verify folder structure with CORRECT paths
print("\n📋 Verifying Dataset Structure:")

# These are the ACTUAL folder paths
required_folders = {
    'data/train/images': 'Training images',
    'data/train/labels': 'Training labels',
    'data/valid/images': 'Validation images',
    'data/valid/labels': 'Validation labels'
}

all_exist = True
for folder, description in required_folders.items():
    exists = os.path.exists(folder)
    status = "✅" if exists else "❌"
    print(f"   {status} {folder:<25} ({description})")
    
    if not exists:
        all_exist = False
        print(f"      Missing: {os.path.abspath(folder)}")

if not all_exist:
    print("\n❌ Some folders are missing!")
    print("\n💡 Your folder structure should be:")
    print("   Intruder-Detection-System/")
    print("   ├── data/")
    print("   │   ├── train/")
    print("   │   │   ├── images/")
    print("   │   │   └── labels/")
    print("   │   ├── valid/")
    print("   │   │   ├── images/")
    print("   │   │   └── labels/")
    print("   │   └── test/")
    print("   ├── data.yaml")
    print("   └── train_weapon_model.py")
    exit()

# Count images
train_images_path = 'data/train/images'
valid_images_path = 'data/valid/images'

train_images = [f for f in os.listdir(train_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
valid_images = [f for f in os.listdir(valid_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

train_count = len(train_images)
valid_count = len(valid_images)

print(f"\n📊 Dataset Statistics:")
print(f"   Training images: {train_count}")
print(f"   Validation images: {valid_count}")
print(f"   Total: {train_count + valid_count}")

if train_count == 0:
    print("\n❌ No training images found!")
    print(f"   Check: {os.path.abspath(train_images_path)}")
    exit()

if valid_count == 0:
    print("\n❌ No validation images found!")
    print(f"   Check: {os.path.abspath(valid_images_path)}")
    print("\n💡 Run split_dataset.py to create validation set")
    exit()

# Quick label check
train_labels_path = 'data/train/labels'
train_labels = [f for f in os.listdir(train_labels_path) if f.endswith('.txt')]
print(f"\n🏷️  Label files:")
print(f"   Training labels: {len(train_labels)}")

if len(train_labels) == 0:
    print("\n⚠️  No label files found!")
    print("   Training may fail without labels")
    choice = input("\n   Continue anyway? (y/n): ")
    if choice.lower() != 'y':
        exit()

# Show sample label
if train_labels:
    sample_label = os.path.join(train_labels_path, train_labels[0])
    print(f"\n📄 Sample label file: {train_labels[0]}")
    try:
        with open(sample_label, 'r') as f:
            content = f.read().strip()
            if content:
                print(f"   Content: {content}")
            else:
                print("   ⚠️  Label file is EMPTY!")
    except:
        pass

# Load model
print("\n🔄 Loading YOLOv8n base model...")
try:
    model = YOLO('yolov8n.pt')
    print("✅ Base model loaded!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Training configuration
print("\n⚙️  Training Configuration:")
print("   Model: YOLOv8n")
print("   Epochs: 30 (faster training)")
print("   Batch: 4")
print("   Image size: 640x640")
print("   Device: CPU")
print("   Dataset: data/train & data/valid")

print("\n⏱️  Estimated time: 30-45 minutes (CPU)")

input("\nPress ENTER to start training (Ctrl+C to cancel)...")

print("\n" + "="*60)
print("🔥 TRAINING STARTED")
print("="*60 + "\n")

try:
    results = model.train(
        data='data.yaml',
        epochs=30,
        imgsz=640,
        batch=4,
        name='weapon_final',
        patience=10,
        save=True,
        save_period=5,
        plots=True,
        device='cpu',
        workers=2,
        verbose=True,
        exist_ok=True,
        cache=False
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    
    # Find and copy model
    model_dir = "runs/detect/weapon_final"
    weights_dir = os.path.join(model_dir, "weights")
    
    print(f"\n📁 Checking for trained model...")
    print(f"   Location: {weights_dir}")
    
    model_copied = False
    
    if os.path.exists(weights_dir):
        weight_files = os.listdir(weights_dir)
        print(f"   Files found: {weight_files}")
        
        # Try best.pt first
        best_path = os.path.join(weights_dir, "best.pt")
        last_path = os.path.join(weights_dir, "last.pt")
        
        if os.path.exists(best_path):
            shutil.copy(best_path, 'best.pt')
            size = os.path.getsize('best.pt') / (1024*1024)
            print(f"\n✅ SUCCESS! Model copied: best.pt ({size:.1f} MB)")
            model_copied = True
            
        elif os.path.exists(last_path):
            shutil.copy(last_path, 'best.pt')
            size = os.path.getsize('best.pt') / (1024*1024)
            print(f"\n✅ SUCCESS! Using last.pt as best.pt ({size:.1f} MB)")
            model_copied = True
            
        else:
            print(f"\n⚠️  No .pt files found in weights folder!")
    else:
        print(f"\n❌ Weights folder not created!")
    
    # Validate if model exists
    if model_copied and os.path.exists('best.pt'):
        print("\n🔍 Validating model...")
        try:
            test_model = YOLO('best.pt')
            metrics = test_model.val(data='data.yaml')
            
            print(f"\n📈 Model Performance:")
            print(f"   Precision: {metrics.box.mp*100:.1f}%")
            print(f"   Recall: {metrics.box.mr*100:.1f}%")
            print(f"   mAP50: {metrics.box.map50*100:.1f}%")
            print(f"   mAP50-95: {metrics.box.map*100:.1f}%")
            
            if metrics.box.map50 > 0.8:
                print("\n✅ Excellent model! Ready to use")
            elif metrics.box.map50 > 0.7:
                print("\n✅ Good model! Suitable for testing")
            else:
                print("\n⚠️  Model accuracy is low. May need more training")
                
        except Exception as e:
            print(f"   ⚠️  Validation error: {e}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    
    if model_copied:
        print("\n✅ Model ready: best.pt")
        print("\n🚀 Next steps:")
        print("   1. Check results: runs/detect/weapon_final/")
        print("   2. Run detection: python intruder_detect_withWeapon.py")
    else:
        print("\n⚠️  Model not found!")
        print("   Check runs/detect/weapon_final/weights/ manually")
    
    print("\n" + "="*60)

except KeyboardInterrupt:
    print("\n\n⚠️  Training cancelled by user")

except Exception as e:
    print("\n" + "="*60)
    print("❌ TRAINING FAILED")
    print("="*60)
    print(f"\nError: {str(e)}")
    
    print("\n💡 Common solutions:")
    print("   1. Reduce batch size:")
    print("      Change batch=4 to batch=2")
    
    print("\n   2. Check dataset:")
    print("      - Images are valid (not corrupted)")
    print("      - Labels exist and match")
    print("      - Labels are in YOLO format")
    
    print("\n   3. Free up disk space:")
    print("      Training needs ~500MB free")
    
    print("\n   4. Check label format:")
    print("      Open data/train/labels/*.txt")
    print("      Should be: class x y w h")
    print("      Example: 0 0.5 0.5 0.2 0.3")
    
    import traceback
    print("\n🔍 Full error trace:")
    traceback.print_exc()