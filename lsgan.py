import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets.load_medmnist import get_medmnist_loaders
from utils.visualize import save_sample_images
from utils.metrics import compute_fid, compute_inception_score, log_generated_images

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDMNIST_VERBOSE"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28
latent_dim = 100
channels = 3
epochs = 50
batch_size = 64
sample_dir = "generated_images/lsgan"
os.makedirs(sample_dir, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), channels, img_size, img_size)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * img_size * img_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        return self.model(img)

def train():
    train_loader, _ = get_medmnist_loaders(batch_size)
    G = Generator().to(device)
    D = Discriminator().to(device)

    adversarial_loss = nn.MSELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

    writer = SummaryWriter(log_dir="runs/lsgan")

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            real = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = G(z)
            loss_G = adversarial_loss(D(gen_imgs), real)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            real_loss = adversarial_loss(D(imgs), real)
            fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        save_sample_images(G, sample_dir, device)
        writer.add_scalar("Loss/Generator", loss_G.item(), epoch)
        writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch)
        log_generated_images(G, writer, tag="Generated_Final", device=device)

    compute_inception_score(G, device, writer=writer, name="LSGAN")
    compute_fid(G, device, writer=writer, name="LSGAN")
    writer.close()

if __name__ == "__main__":
    train()
