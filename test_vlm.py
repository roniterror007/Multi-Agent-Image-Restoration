from PIL import Image
import numpy as np
from core.degradation_classifier import DegradationReport, DegradationType
from core.vlm_reasoner import VLMReasoner
from core.agent_controller import validate_adjust_lab_call

img = Image.new('RGB', (256, 256), color = 'pink')
lab = np.zeros((256, 256, 3), dtype=np.uint8)
report = DegradationReport(primary=DegradationType.PINK_CAST, severity=0.8, metrics={'mean_a': 150.0})

vlm = VLMReasoner()
payload = vlm.analyze(img, lab, "123", report)
print("Payload:", payload)
print("Validated:", validate_adjust_lab_call(payload))
