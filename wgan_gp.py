import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from datasets.load_medmnist import get_medmnist_loaders
from utils.metrics import compute_inception_score, compute_fid, log_generated_images
from utils.visualize import save_sample_images

# Hyperparameters
latent_dim = 100
image_size = 28
image_channels = 3
batch_size = 64
lambda_gp = 10
n_critic = 5
epochs = 50
sample_dir = "generated_images/wgan_gp"
log_dir = "runs/wgan_gp"
os.makedirs(sample_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, image_channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z).view(z.size(0), image_channels, image_size, image_size)
        return img

# Critic
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_channels * image_size * image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        return self.model(img)

def compute_gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    validity = critic(interpolated)
    ones = torch.ones_like(validity).to(device)
    gradients = torch.autograd.grad(outputs=validity, inputs=interpolated, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(real.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Training
def train():
    G = Generator().to(device)
    D = Critic().to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    train_loader, _ = get_medmnist_loaders(batch_size=batch_size)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            for _ in range(n_critic):
                z = torch.randn(bs, latent_dim).to(device)
                fake_imgs = G(z).detach()
                d_real = D(real_imgs)
                d_fake = D(fake_imgs)
                gp = compute_gradient_penalty(D, real_imgs, fake_imgs)
                d_loss = -torch.mean(d_real) + torch.mean(d_fake) + lambda_gp * gp
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            if i % n_critic == 0:
                z = torch.randn(bs, latent_dim).to(device)
                gen_imgs = G(z)
                g_loss = -torch.mean(D(gen_imgs))
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

                print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        save_sample_images(G, sample_dir, device)
        writer.add_scalar("Loss/Discriminator", d_loss.item(), epoch)
        writer.add_scalar("Loss/Generator", g_loss.item(), epoch)
        log_generated_images(G, writer, tag="Generated_Final", device=device)

    compute_inception_score(G, device, writer=writer, name="WGAN-GP")
    compute_fid(G, device, writer=writer, name="WGAN-GP")
    writer.close()

if __name__ == "__main__":
    train()
