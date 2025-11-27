import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np
from tqdm import tqdm

##################################
# Hyperparameters
##################################
latent_dim = 100
num_classes = 10       # digits 0 through 9
batch_size = 256
epochs = 100
learning_rate = 0.0002

# Make sure to use GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create folder to (optionally) store sample images
os.makedirs("cgan_samples", exist_ok=True)

##################################
# Data Loading & Transform
##################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Range: [-1, 1]
])

train_dataset = datasets.MNIST(root="mnist_data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

##################################
# One-Hot Label Utility
##################################
def label_to_onehot(labels, num_classes):
    """
    Convert a batch of integer labels to one-hot encoding.
    labels: shape (batch_size,)
    returns: shape (batch_size, num_classes)
    """
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot[range(batch_size), labels] = 1.0
    return one_hot

##################################
# Class-Conditional Generator
##################################
# Input: noise (latent_dim) + one-hot label (num_classes)
# Output: 28x28 image
##################################
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        input_dim = latent_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, noise, labels_onehot):
        x = torch.cat([noise, labels_onehot], dim=1)  # (batch_size, latent_dim + num_classes)
        out = self.model(x)
        out = out.view(out.size(0), 1, 28, 28)
        return out

##################################
# Class-Conditional Discriminator
##################################
# Input: image (28x28) + one-hot label
# Output: Real/Fake probability
##################################
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        input_dim = 28*28 + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels_onehot):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat([img_flat, labels_onehot], dim=1)
        out = self.model(x)
        return out

##################################
# Initialize Networks
##################################
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

##################################
# Training Loop
##################################
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        # Real images
        batch_size_real = imgs.size(0)
        real_imgs = imgs.to(device)
        real_labels = labels.to(device)

        # One-hot encode the real labels
        real_labels_oh = label_to_onehot(real_labels, num_classes)

        valid = torch.ones(batch_size_real, 1, device=device)
        fake = torch.zeros(batch_size_real, 1, device=device)

        # ---------------------------
        #  Train Generator
        # ---------------------------
        optimizer_G.zero_grad()

        # Sample random noise and random labels
        z = torch.randn(batch_size_real, latent_dim, device=device)
        gen_labels_int = torch.randint(0, num_classes, (batch_size_real,), device=device)
        gen_labels_oh = label_to_onehot(gen_labels_int, num_classes)

        # Generate images
        gen_imgs = generator(z, gen_labels_oh)

        # Loss: discriminator wants these to be real
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels_oh), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------------
        #  Train Discriminator
        # ---------------------------
        optimizer_D.zero_grad()

        # Discriminator loss on real images
        real_loss = adversarial_loss(discriminator(real_imgs, real_labels_oh), valid)

        # Discriminator loss on fake images
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels_oh), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

##################################
# Save Model Weights
##################################
# This enables using the trained models later in another program.
##################################
model_folder = "cgan_models"
os.makedirs(model_folder, exist_ok=True)

torch.save(generator.state_dict(), os.path.join(model_folder, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(model_folder, "discriminator.pth"))

print(f"Generator and Discriminator weights saved to '{model_folder}' folder.")