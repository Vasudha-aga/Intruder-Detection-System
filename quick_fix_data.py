"""
Quick Fix: Create correct data.yaml file
Run this first, then run train_weapon_model.py
"""

import os

print("🔧 Fixing data.yaml configuration...")

# Delete old 'data' file if exists
if os.path.exists('data'):
    try:
        os.remove('data')
        print("✅ Removed old 'data' file")
    except:
        print("⚠️  Could not delete 'data' file (might be in use)")

# Create correct data.yaml
yaml_content = """# YOLOv8 Weapon Detection Dataset Configuration
# Dataset: Weapon Detector Computer Vision v3
# Images: 6788
# Classes: weapon

path: .
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['weapon']
"""

with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print("✅ Created data.yaml with correct format")

# Verify dataset structure
print("\n📋 Checking dataset structure...")

folders = {
    'train/images': 0,
    'train/labels': 0,
    'valid/images': 0,
    'valid/labels': 0
}

for folder in folders.keys():
    if os.path.exists(folder):
        files = os.listdir(folder)
        if 'images' in folder:
            count = len([f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            count = len([f for f in files if f.endswith('.txt')])
        folders[folder] = count
        print(f"   ✅ {folder}: {count} files")
    else:
        print(f"   ❌ {folder}: NOT FOUND!")

# Summary
train_imgs = folders['train/images']
train_lbls = folders['train/labels']
valid_imgs = folders['valid/images']
valid_lbls = folders['valid/labels']

print(f"\n📊 Summary:")
print(f"   Training: {train_imgs} images, {train_lbls} labels")
print(f"   Validation: {valid_imgs} images, {valid_lbls} labels")
print(f"   Total: {train_imgs + valid_imgs} images")

if train_imgs == train_lbls and valid_imgs == valid_lbls:
    print("\n✅ Dataset structure looks good!")
    print("\n🚀 Next step: Run training")
    print("   python train_weapon_model.py")
else:
    print("\n⚠️  Warning: Image and label counts don't match")
    print("   This might cause issues during training")

print("\n" + "="*60)