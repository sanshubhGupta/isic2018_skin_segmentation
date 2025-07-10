import gradio as gr
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("best_unet.pth", map_location=device))
model.eval().to(device)

# Preprocessing
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

def predict(image):
    image_np = np.array(image.convert("RGB"))
    aug = transform(image=image_np)
    input_tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))[0,0].cpu().numpy()
    mask = (output > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Skin Lesion Segmentation (ISIC 2018)",
    description="Upload a dermoscopic image and receive the predicted lesion mask."
).launch()
