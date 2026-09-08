import os
from PIL import Image, ImageDraw, ImageFilter
import random
import math

out_dir = "images"
os.makedirs(out_dir, exist_ok=True)

def draw_gear(draw, cx, cy, radius, teeth):
    for i in range(360):
        r = radius + (10 if math.sin(math.radians(i * teeth)) > 0 else -10)
        x = cx + r * math.cos(math.radians(i))
        y = cy + r * math.sin(math.radians(i))
        draw.point((x, y), fill=(100, 100, 100))
    draw.ellipse((cx-radius/2, cy-radius/2, cx+radius/2, cy+radius/2), fill=(50, 50, 50), outline=(100,100,100))

def generate_procedural_part(name, index):
    img = Image.new("RGB", (512, 512), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add some noise
    for _ in range(1000):
        draw.point((random.randint(0,511), random.randint(0,511)), fill=(200,200,200))
        
    color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
    
    if index % 3 == 0:
        # Draw abstract PCB-like
        draw.rectangle((100, 100, 412, 412), fill=(34, 139, 34))
        for _ in range(20):
            draw.line((random.randint(100, 412), random.randint(100, 412), random.randint(100, 412), random.randint(100, 412)), fill=(218, 165, 32), width=3)
        draw.rectangle((200, 200, 300, 300), fill=(20, 20, 20))
    elif index % 3 == 1:
        # Draw metallic block
        draw.rectangle((150, 150, 362, 362), fill=color)
        draw.ellipse((200, 200, 312, 312), fill=(30, 30, 30))
        draw.rectangle((240, 100, 272, 412), fill=(150, 150, 150))
    else:
        # Draw gear/round part
        draw_gear(draw, 256, 256, 150, 12)
        
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(os.path.join(out_dir, f"procedural_part_{name}.jpg"))

for i in range(15):
    generate_procedural_part(f"00{i}", i)

print("Generated 15 procedural parts.")
