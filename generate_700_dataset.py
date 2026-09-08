import os
import random
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path

base_dir = Path(__file__).resolve().parent
good_pics_dir = base_dir / "good_pics"
images_dir = base_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

base_images = list(good_pics_dir.glob("*.png"))
if not base_images:
    print("No base images found in good_pics/")
    exit(1)

def apply_heavy_degradation(img):
    img = img.convert("RGB")
    
    # Brightness (0.3 to 1.7)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.3, 1.7))
    
    # Color balance/cast
    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * random.uniform(0.7, 1.3))))
    g = g.point(lambda i: min(255, int(i * random.uniform(0.7, 1.3))))
    b = b.point(lambda i: min(255, int(i * random.uniform(0.7, 1.3))))
    img = Image.merge("RGB", (r, g, b))
    
    # Saturation
    img = ImageEnhance.Color(img).enhance(random.uniform(0.1, 2.0))
    
    # Occasional slight blur
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
    return img

total_to_generate = 700
generated = 0

print(f"Generating {total_to_generate} synthetic samples...")
while generated < total_to_generate:
    for base_path in base_images:
        if generated >= total_to_generate:
            break
            
        img = Image.open(base_path)
        aug_img = apply_heavy_degradation(img.copy())
        
        out_name = f"synth_{generated:04d}_{base_path.stem}.png"
        aug_img.save(images_dir / out_name)
        generated += 1
        
        if generated % 100 == 0:
            print(f"Generated {generated}/{total_to_generate}...")

print("Dataset synthesis complete!")
