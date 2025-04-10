import torch
from torchvision import transforms, utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.tensorboard import SummaryWriter
import os

# Convert [-1, 1] images → [0, 255] uint8 + grayscale to RGB
def preprocess_for_metrics(images):
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = (images * 255).to(torch.uint8)
    return images

# Save image grid to TensorBoard
def log_generated_images(generator, writer, tag, device):
    z = torch.randn(64, 100).to(device)
    generator.eval()
    with torch.no_grad():
        samples = generator(z)
    samples_grid = vutils.make_grid(samples, normalize=True, scale_each=True)
    writer.add_image(tag, samples_grid, global_step=0)

# ✅ INCEPTION SCORE
def compute_inception_score(generator, device, writer=None, name="Model"):
    transform = transforms.Resize((299, 299))
    inception = InceptionScore(normalize=True).to(device)

    generator.eval()
    with torch.no_grad():
        for _ in range(10):
            z = torch.randn(64, 100).to(device)
            fake_images = generator(z)
            fake_images = transform(fake_images)
            fake_images = preprocess_for_metrics(fake_images)
            inception.update(fake_images)

    score, std = inception.compute()
    print(f"[{name}] Inception Score: {score:.4f} ± {std:.4f}")
    if writer:
        writer.add_scalar(f"{name}/Inception_Score", score, 0)
    generator.train()

# ✅ FID SCORE
def compute_fid(generator, device, writer=None, name="Model"):
    from datasets.load_medmnist import get_medmnist_loaders
    real_loader, _ = get_medmnist_loaders(batch_size=64)

    transform = transforms.Resize((299, 299))
    fid = FrechetInceptionDistance(feature=64).to(device)

    generator.eval()
    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(real_loader):
            if i > 10: break
            real_imgs = transform(real_imgs.to(device))
            real_imgs = preprocess_for_metrics(real_imgs)
            fid.update(real_imgs, real=True)

        for _ in range(10):
            z = torch.randn(64, 100).to(device)
            fake_imgs = generator(z)
            fake_imgs = transform(fake_imgs)
            fake_imgs = preprocess_for_metrics(fake_imgs)
            fid.update(fake_imgs, real=False)

    score = fid.compute()
    print(f"[{name}] FID Score: {score:.4f}")
    if writer:
        writer.add_scalar(f"{name}/FID_Score", score, 0)
    generator.train()
