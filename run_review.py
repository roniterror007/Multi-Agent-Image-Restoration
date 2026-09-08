import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from core.review_ui import review_image
from core.feedback_store import compute_perceptual_hash
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="images")
    parser.add_argument("--output-dir", default="abb_output_full") # Align with run_pipeline.py
    parser.add_argument("--feedback-dir", default="feedback_auto")
    parser.add_argument("--image", default=None, help="Specific image to review")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    feedback_dir = Path(args.feedback_dir).resolve()
    feedback_dir.mkdir(exist_ok=True, parents=True)

    overrides_file = feedback_dir / "image_overrides.json"
    overrides = {}
    if overrides_file.exists():
        with open(overrides_file, "r") as f:
            try:
                overrides = json.load(f)
            except:
                pass

    if args.image:
        images_to_review = [output_dir / args.image]
    else:
        images_to_review = list(output_dir.glob("*.*"))

    for out_img in images_to_review:
        if out_img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
            
        in_img = input_dir / out_img.name
        if not in_img.exists():
            continue

        raw = Image.open(in_img).convert("RGB")
        cleaned = Image.open(out_img).convert("RGB")

        print(f"\n==============================================")
        print(f"Reviewing {out_img.name}...")
        
        result = review_image(
            original=raw,
            cleaned=cleaned,
            filename=out_img.name,
            feedback_dir=str(feedback_dir),
            route="review"
        )
        
        decision = result.get('decision')
        print(f"Result for {out_img.name}: {decision}")
        
        if decision == 'kill':
            print("Pipeline explicitly killed by user from UI.")
            sys.exit(0)
            
        elif decision == 'manual_review':
            print(f"[{out_img.name}] Escalated to manual offline review.")
            
        elif decision == 'accepted':
            print(f"[{out_img.name}] Accepted.")
            if result.get('is_custom', False):
                shifts = result['shifts']
                phash = compute_perceptual_hash(raw)
                overrides[phash] = shifts
                with open(overrides_file, "w") as f:
                    json.dump(overrides, f, indent=4)
                print(f"Saved custom shifts {shifts} for pHash {phash} to image_overrides.json")

if __name__ == "__main__":
    main()
