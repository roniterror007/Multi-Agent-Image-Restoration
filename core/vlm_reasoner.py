import os
import torch
from PIL import Image
from core.degradation_classifier import DegradationReport

class VLMReasoner:
    """
    True Vision-Language Model interface running locally via HuggingFace.
    Uses 'Salesforce/blip-image-captioning-base' to analyze lighting and color.
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
    def _load_model(self):
        if self.model is not None:
            return
            
        print("[VLM Reasoner] Loading BLIP Vision Model from HuggingFace...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            self.model.eval()
            print("[VLM Reasoner] BLIP loaded successfully!")
        except Exception as e:
            print(f"[VLM Reasoner] Failed to load BLIP: {e}")
            self.model = "failed"

    def analyze(self, img_pil: Image.Image, img_lab, phash: str, report: DegradationReport) -> dict:
        self._load_model()
        
        if self.model == "failed" or self.model is None:
            return self._fallback_expert_system(report)
            
        try:
            prompt = "describe the overall lighting and color balance of this image:"
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Clear fragmented memory before generation
                
                inputs = self.processor(img_pil, prompt, return_tensors="pt").to(self.device, dtype=self.model.dtype if self.device.type != 'cpu' else None)
                # Ensure pixel_values is correctly casted
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
                    
                out = self.model.generate(**inputs, max_new_tokens=15)
                response = self.processor.decode(out[0], skip_special_tokens=True).lower()
                
            return self._parse_vlm_response(response, report)
            
        except Exception as e:
            print(f"[VLM Reasoner] Error during inference: {e}")
            return self._fallback_expert_system(report)
            
    def _parse_vlm_response(self, text: str, report: DegradationReport) -> dict:
        """Translates semantic English from BLIP into numerical Lab shifts."""
        reasons = [f"BLIP Output: '{text}'"]
        dl, da, db = 0, 0, 0
        confidence = 0.80
        
        # Parse L (Lightness)
        if "dark" in text or "underexpos" in text or "black" in text:
            dl = 20
            reasons.append("VLM parsed: Increasing brightness (+20 L).")
            confidence += 0.05
        elif "bright" in text or "white" in text or "overexpos" in text:
            dl = -20
            reasons.append("VLM parsed: Decreasing brightness (-20 L).")
            confidence += 0.05
            
        # Parse A (Green-Red)
        if "pink" in text or "red" in text or "magenta" in text:
            da = -15
            reasons.append("VLM parsed: Neutralizing pink cast (-15 a*).")
            confidence += 0.05
        elif "green" in text:
            da = 15
            reasons.append("VLM parsed: Neutralizing green cast (+15 a*).")
            confidence += 0.05
            
        # Parse B (Blue-Yellow)
        if "yellow" in text or "orange" in text or "brown" in text:
            db = -15
            reasons.append("VLM parsed: Neutralizing warm cast (-15 b*).")
            confidence += 0.05
        elif "blue" in text or "cool" in text:
            db = 15
            reasons.append("VLM parsed: Neutralizing cool cast (+15 b*).")
            confidence += 0.05
            
        if dl == 0 and da == 0 and db == 0:
            reasons.append("VLM did not detect strong degradation. Minor contrast bump applied.")
            dl = 5
            
        return {
            "action": "adjust_color",
            "delta_l_ticks": dl,
            "delta_a_ticks": da,
            "delta_b_ticks": db,
            "strength_ticks": 10,
            "reasoning": " ".join(reasons),
            "confidence": min(0.96, confidence)
        }
        
    def _fallback_expert_system(self, report: DegradationReport) -> dict:
        """Rigorous mathematical expert system fallback."""
        mean_l = report.metrics.get("mean_l", 128.0)
        mean_a = report.metrics.get("mean_a", 128.0)
        mean_b = report.metrics.get("mean_b", 128.0)
        
        reasons = []
        dl, da, db = 0, 0, 0
        confidence = 0.70
        
        if report.has_exposure_issue:
            target_l = 150.0
            dl = int((target_l - mean_l) * 0.75)
            reasons.append(f"Expert System: Exposure fix (L shift {dl}).")
            confidence += 0.10
            
        if report.is_bw:
            reasons.append("Expert System: Monochrome detected.")
            da, db = 0, 0
            confidence += 0.20
        elif report.has_color_cast:
            da = int((128.0 - mean_a) * 0.85)
            db = int((128.0 - mean_b) * 0.85)
            reasons.append(f"Expert System: Neutralization (da={da}, db={db}).")
            confidence += 0.15
            
        if not reasons:
            reasons.append("Expert System: Image within prior bounds.")
            dl = 1
            
        dl = max(-50, min(50, dl))
        da = max(-50, min(50, da))
        db = max(-50, min(50, db))
        
        return {
            "action": "adjust_color",
            "delta_l_ticks": dl,
            "delta_a_ticks": da,
            "delta_b_ticks": db,
            "strength_ticks": 10,
            "reasoning": " ".join(reasons),
            "confidence": min(0.95, confidence)
        }
