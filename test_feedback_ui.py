import os
import shutil
import json
from pathlib import Path
from PIL import Image
from core.feedback_store import compute_perceptual_hash

# 1. Setup paths
base_dir = Path(__file__).resolve().parent
test_in = base_dir / "test_feedback_in"
test_out = base_dir / "test_feedback_out"
test_feed = base_dir / "test_feedback_store"

for d in [test_in, test_out, test_feed]:
    d.mkdir(exist_ok=True, parents=True)

# 2. Pick an image
src_image = base_dir / "bad_images" / "part_gear_1788690772476_pink.png"
dst_image = test_in / src_image.name
shutil.copy(src_image, dst_image)

# 3. Compute pHash
img_pil = Image.open(dst_image).convert("RGB")
phash = compute_perceptual_hash(img_pil)
print(f"Computed pHash for test image: {phash}")

# 4. Create mock override (simulate user saying "shift L by +10, a by -40, b by +5")
override_data = {
    phash: [10, -40, 5]
}
with open(test_feed / "image_overrides.json", "w") as f:
    json.dump(override_data, f)
print("Saved mock manual override.")

# 5. Run pipeline on this single image
import subprocess
print("\nRunning pipeline...")
cmd = [
    "python", "run_pipeline.py",
    "--input-dir", str(test_in),
    "--output-dir", str(test_out),
    "--feedback-dir", str(test_feed),
]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)
