import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing Moondream2 Load with padding fix...")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        revision=revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded successfully!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    # FIX: Provide a pad_token_id to avoid generation crashes
    if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Create dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    
    enc_image = model.encode_image(img)
    response = model.answer_question(enc_image, "What color is this image?", tokenizer)
    print("Response:", response)
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
