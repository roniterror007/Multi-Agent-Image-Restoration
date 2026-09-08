# xai/birefnet_tensor_isolation.py
import numpy as np
from PIL import Image, ImageDraw


class IsolatedBiRefNetProcessor:
    def __init__(self, birefnet_model):
        self.model = birefnet_model
        self.clean_tensor = None
        self.viz_pil = None
        self.original_size = None

    def set_clean_image(self, image_pil: Image.Image):
        """Locks in the pure image for machine inference."""
        self.original_size = image_pil.size
        self.viz_pil = image_pil.copy()

        # Convert to clean tensor for BiRefNet (No UI elements exist here)
        import torchvision.transforms.functional as F

        tensor = F.to_tensor(image_pil).unsqueeze(0)
        self.clean_tensor = F.resize(tensor, [1024, 1024], antialias=True)

    def add_visualization_box(self, box: tuple[int, int, int, int], color="red"):
        """Draws UI elements ONLY on the display layer, protecting the inference tensor."""
        if self.viz_pil is None:
            raise ValueError("Clean image must be set before adding visualizations.")

        draw = ImageDraw.Draw(self.viz_pil)
        draw.rectangle(box, outline=color, width=4)
        return self.viz_pil

    def segment(self):
        """Executes segmentation strictly on the uncontaminated tensor."""
        if self.clean_tensor is None:
            raise ValueError("No clean tensor available for segmentation.")

        if self.model is None:
            # Fallback if no model loaded
            dummy_mask = Image.new("L", self.original_size, color=255)
            return dummy_mask

        import torch
        device = next(self.model.parameters()).device
        
        # Normalize as expected by BiRefNet
        input_tensor = self.clean_tensor.to(device)
        model_dtype = next(self.model.parameters()).dtype
        input_tensor = input_tensor.to(dtype=model_dtype)
        
        input_tensor = input_tensor / 255.0 if input_tensor.max() > 1.0 else input_tensor
        # Mean/std normalization (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device, dtype=model_dtype)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device, dtype=model_dtype)
        input_tensor = (input_tensor - mean) / std

        with torch.no_grad():
            preds = self.model(input_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            
            # Apply sigmoid because BiRefNet outputs logits
            preds = preds.sigmoid().squeeze().cpu().float()
            
            import torchvision.transforms.functional as F
            from PIL import Image
            
            # Resize mask back to original size
            mask_pil = F.to_pil_image(preds)
            mask_pil = mask_pil.resize(self.original_size, Image.Resampling.BILINEAR)
            
            return mask_pil

