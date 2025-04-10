import torch
import torchvision.utils as vutils
import os

def save_sample_images(generator, out_dir, device):
    z = torch.randn(64, 100).to(device)
    with torch.no_grad():
        samples = generator(z).detach().cpu()
    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(samples, os.path.join(out_dir, "generated_sample.png"), normalize=True)
    print(f"Sample image saved to {os.path.join(out_dir, 'generated_sample.png')}")
