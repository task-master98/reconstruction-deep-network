from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import torch

model_mapping = {
    "vae": AutoencoderKL,
    "scheduler": DDIMScheduler,
    "unet": UNet2DConditionModel,
    "tokenizer": CLIPTokenizer,
    "text_encoder": CLIPTextModel
}

def load_pretrained_model_img(model_id: str, model_type: str):
    assert model_type in model_mapping
    model = model_mapping[model_type].from_pretrained(model_id, subfolder=model_type)
    if model_type == "vae":
        model.eval()
    
    
    return model

def load_pretrained_model_text(model_id: str, model_type: str):
    assert model_type in model_mapping
    model = model_mapping[model_type].from_pretrained(model_id, subfolder=model_type, torch_dtype=torch.float32)
    return model



