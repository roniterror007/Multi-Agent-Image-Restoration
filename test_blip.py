import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Testing BLIP Load...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
    print("BLIP loaded successfully!")
    
    img = Image.new('RGB', (224, 224), color = 'red')
    inputs = processor(img, "the lighting in this image is", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    out = model.generate(**inputs, max_new_tokens=20)
    print("Response:", processor.decode(out[0], skip_special_tokens=True))
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
