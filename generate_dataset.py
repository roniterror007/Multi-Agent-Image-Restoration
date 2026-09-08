import os
import shutil
import random
from PIL import Image, ImageEnhance
from pathlib import Path

# User uploaded files
uploaded_files = [
    r"C:\Users\Ronit\.gemini\antigravity\brain\215eccc7-bc8d-4d24-a999-d6623e592852\.user_uploaded\media_1788683203586.png",
    r"C:\Users\Ronit\.gemini\antigravity\brain\215eccc7-bc8d-4d24-a999-d6623e592852\.user_uploaded\media_1788683221800.png",
    r"C:\Users\Ronit\.gemini\antigravity\brain\215eccc7-bc8d-4d24-a999-d6623e592852\.user_uploaded\media_1788683234760.png"
]

base_dir = Path(__file__).resolve().parent
images_dir = base_dir / "images"
good_pics_dir = base_dir / "good_pics"

# Create target directories
images_dir.mkdir(parents=True, exist_ok=True)
good_pics_dir.mkdir(parents=True, exist_ok=True)
(base_dir / "data" / "images").mkdir(parents=True, exist_ok=True)
(base_dir / "data" / "good_pics").mkdir(parents=True, exist_ok=True)

def apply_random_degradation(img):
    img = img.convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * random.uniform(0.8, 1.2))))
    b = b.point(lambda i: min(255, int(i * random.uniform(0.8, 1.2))))
    img = Image.merge("RGB", (r, g, b))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.2, 1.8))
    return img

print("Processing uploaded images...")
for i, path in enumerate(uploaded_files):
    if not os.path.exists(path):
        continue
    try:
        img = Image.open(path)
        good_name = f"ref_part_{i}.png"
        img.save(good_pics_dir / good_name)
        img.save(base_dir / "data" / "good_pics" / good_name)
        img.save(images_dir / f"input_part_{i}.png")
        for j in range(3):
            aug_img = apply_random_degradation(img.copy())
            aug_img.save(images_dir / f"aug_part_{i}_{j}.png")
        print(f"Generated dataset from {os.path.basename(path)}")
    except Exception as e:
        print(f"Error processing {path}: {e}")

print("Dataset generation complete.")
