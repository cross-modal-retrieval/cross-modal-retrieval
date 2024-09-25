import torch
import clip
from PIL import Image

# Load pre-trained CLIP model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.float()

def encode_image(image_path):
    # Open image and preprocess
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    # Encode image with CLIP model
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Return feature tensor
    return image_features

def get_patch_tokens(image_path):
    # Open image and preprocess
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    # Ensure image has the same dtype as model weights
    dtype = model.visual.conv1.weight.dtype
    image = image.to(dtype)

    # Get patch tokens
    with torch.no_grad():
        x = model.visual.conv1(image)  # shape [1, 768, 7, 7] for ViT-B/32
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # shape [1, 49, 768]
        x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                              device=x.device), x],
                      dim=1)  # Add class token
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)

        for layer in model.visual.transformer.resblocks:
            x = layer(x)

        patch_tokens = x[:, 1:, :]  # Exclude the class token
    return patch_tokens

# Example: Encode a jpg image
if __name__ == '__main__':
    image_path = "/your_path/to/camera_ffd2cca53bfe4dc580469bc1fffb077e_office_22_frame_equirectangular_domain_rgb.jpg"
    features = get_patch_tokens(image_path)
    print(features.shape)
