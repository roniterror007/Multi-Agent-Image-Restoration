import os
import shutil
import numpy as np
import cv2
from pathlib import Path

base_dir = Path(__file__).resolve().parent
images_dir = base_dir / "images"
good_pics_dir = base_dir / "good_pics"
bad_images_dir = base_dir / "bad_images"

good_pics_dir.mkdir(exist_ok=True)
bad_images_dir.mkdir(exist_ok=True)

# 1. Promote all images to good_pics as ground truth
for img_path in images_dir.glob("*.jpg"):
    shutil.copy(img_path, good_pics_dir / img_path.name)
for img_path in images_dir.glob("*.png"):
    shutil.copy(img_path, good_pics_dir / img_path.name)

print(f"Copied all original parts to good_pics/ as ground truth.")

def apply_bad_filters(img_path, out_dir):
    img = cv2.imread(str(img_path))
    if img is None: return
    
    stem = img_path.stem
    
    # 1. Black and White
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / f"{stem}_bw.png"), bw_bgr)
    
    # 2. Overexposed
    # Convert to float to prevent wrapping
    img_f = img.astype(np.float32)
    over = np.clip(img_f * 1.5 + 50, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_over.png"), over)
    
    # 3. Washed out (Low contrast, high brightness)
    washed = np.clip(img_f * 0.5 + 100, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_washed.png"), washed)
    
    # 4. Pinkish Cast
    pink = img_f.copy()
    pink[:, :, 2] = np.clip(pink[:, :, 2] + 60, 0, 255) # Add Red
    pink[:, :, 0] = np.clip(pink[:, :, 0] + 40, 0, 255) # Add Blue
    cv2.imwrite(str(out_dir / f"{stem}_pink.png"), pink.astype(np.uint8))

for img_path in good_pics_dir.glob("*.*"):
    apply_bad_filters(img_path, bad_images_dir)

print(f"Generated bad versions in bad_images/ directory.")
