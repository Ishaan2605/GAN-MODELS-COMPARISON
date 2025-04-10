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
lr = 0.00005
clip_value = 0.01
n_critic = 5
epochs = 50
batch_size = 64
img_shape = (3, 28, 28)
sample_dir = "generated_images/wgan"
log_dir = "runs/wgan"
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
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z).view(z.size(0), *img_shape)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.model(flat)

# Models & Optimizers
G = Generator().to(device)
D = Discriminator().to(device)
opt_G = optim.RMSprop(G.parameters(), lr=lr)
opt_D = optim.RMSprop(D.parameters(), lr=lr)
writer = SummaryWriter(log_dir=log_dir)
train_loader, _ = get_medmnist_loaders(batch_size=batch_size)

# Training
def train():
    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)

            # Discriminator
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = G(z).detach()
            loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Generator every n_critic steps
            if i % n_critic == 0:
                z = torch.randn(real_imgs.size(0), latent_dim).to(device)
                gen_imgs = G(z)
                loss_G = -torch.mean(D(gen_imgs))
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        save_sample_images(G, sample_dir, device)
        writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch)
        writer.add_scalar("Loss/Generator", loss_G.item(), epoch)
        log_generated_images(G, writer, tag="Generated_Final", device=device)

    compute_inception_score(G, device, writer=writer, name="WGAN")
    compute_fid(G, device, writer=writer, name="WGAN")
    writer.close()

if __name__ == "__main__":
    train()
