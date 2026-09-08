import cv2
from core.degradation_classifier import DegradationClassifier

c = DegradationClassifier()

# Test across multiple parts and all degradation types
parts = [
    'part_gear_1788690772476',
    'part_bearing_1788690801913',
    'part_pcb_1788690748374',
    'part_motor_1788690879727',
]

degrades = {
    '_pink': 'pink_cast',
    '_bw': 'grayscale_bw',
    '_washed': 'washed_out',
    '_over': 'overexposed',
}

total = 0
correct = 0

for part in parts:
    for suffix, expected in degrades.items():
        path = f'bad_images/{part}{suffix}.png'
        img = cv2.imread(path)
        if img is None:
            continue
        report = c.classify(img)
        total += 1
        status = "OK" if expected == report.primary.value else "MISS"
        if expected == report.primary.value:
            correct += 1
        sec = [d.value for d in report.secondary]
        print(f"{part[-15:]}{suffix:10s} => {report.primary.value:15s} (expected {expected:15s}) {status} sec={sec}")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
