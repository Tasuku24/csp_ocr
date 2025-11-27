import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

##################################
# Configuration
##################################
class Config:
    latent_dim = 100
    num_classes = 10
    batch_size = 64
    num_samples_per_digit = 5  # Number of real samples to use per digit
    num_epochs = 30
    learning_rate = 0.0001
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

##################################
# Generator and Discriminator Architecture
##################################
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        input_dim = latent_dim + num_classes
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        img = self.model(x)
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        input_dim = 28*28 + num_classes
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat([img_flat, labels], dim=1)
        validity = self.model(x)
        return validity

##################################
# Utility Functions
##################################
def load_random_samples(collected_dataset_path, num_samples_per_digit):
    """Load random samples from collected_dataset folder"""
    samples = {}
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for digit in range(10):
        digit_path = os.path.join(collected_dataset_path, str(digit))
        if not os.path.exists(digit_path):
            raise FileNotFoundError(f"Path not found: {digit_path}")
        
        # Get all image files in the digit folder
        image_files = [f for f in os.listdir(digit_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) < num_samples_per_digit:
            raise ValueError(f"Not enough samples for digit {digit}")
        
        # Randomly select images
        selected_files = random.sample(image_files, num_samples_per_digit)
        
        # Load and transform images
        digit_samples = []
        for img_file in selected_files:
            img_path = os.path.join(digit_path, img_file)
            img = Image.open(img_path)
            img_tensor = transform(img)
            digit_samples.append(img_tensor)
        
        samples[digit] = torch.stack(digit_samples)
    
    return samples

def label_to_onehot(labels, num_classes=10):
    """Convert integer labels to one-hot encoded tensors"""
    batch_size = labels.size(0)
    onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    onehot[range(batch_size), labels] = 1
    return onehot

##################################
# Reinforcement Learning GAN Trainer
##################################
class RLGANTrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize networks
        self.generator = Generator(config.latent_dim, config.num_classes).to(config.device)
        self.discriminator = Discriminator(config.num_classes).to(config.device)
        
        # Load pre-trained weights
        self.generator.load_state_dict(
            torch.load('cgan_models/generator.pth', map_location=config.device)
        )
        self.discriminator.load_state_dict(
            torch.load('cgan_models/discriminator.pth', map_location=config.device)
        )
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=config.learning_rate)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=config.learning_rate)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.similarity_loss = nn.MSELoss()

    def train(self, real_samples):
        """Train the GAN using reinforcement learning principles"""
        for epoch in range(self.config.num_epochs):
            total_g_loss = 0
            total_d_loss = 0
            
            # Train on each digit
            for digit in range(10):
                # Get real samples for this digit
                real_digit_imgs = real_samples[digit].to(self.config.device)
                batch_size = real_digit_imgs.size(0)
                
                # Create labels
                real = torch.ones(batch_size, 1).to(self.config.device)
                fake = torch.zeros(batch_size, 1).to(self.config.device)
                digit_labels = torch.full((batch_size,), digit, dtype=torch.long).to(self.config.device)
                digit_onehot = label_to_onehot(digit_labels, 10)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                
                # Loss on real images
                real_pred = self.discriminator(real_digit_imgs, digit_onehot)
                d_real_loss = self.adversarial_loss(real_pred, real)
                
                # Loss on fake images
                z = torch.randn(batch_size, self.config.latent_dim).to(self.config.device)
                fake_imgs = self.generator(z, digit_onehot)
                fake_pred = self.discriminator(fake_imgs.detach(), digit_onehot)
                d_fake_loss = self.adversarial_loss(fake_pred, fake)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                
                # Generate new images
                z = torch.randn(batch_size, self.config.latent_dim).to(self.config.device)
                gen_imgs = self.generator(z, digit_onehot)
                
                # Adversarial loss
                validity = self.discriminator(gen_imgs, digit_onehot)
                g_adv_loss = self.adversarial_loss(validity, real)
                
                # Similarity loss to real samples (reinforcement signal)
                g_sim_loss = self.similarity_loss(gen_imgs, real_digit_imgs)
                
                # Total generator loss
                g_loss = g_adv_loss + g_sim_loss
                g_loss.backward()
                self.optimizer_G.step()

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()

            # Print epoch statistics
            avg_g_loss = total_g_loss / 10
            avg_d_loss = total_d_loss / 10
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                  f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

            # Save sample images every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_sample_images(epoch + 1)

    def save_sample_images(self, epoch):
        """Generate and save sample images"""
        os.makedirs('rl_samples', exist_ok=True)
        
        # Switch to eval mode for generation
        self.generator.eval()
        
        with torch.no_grad():
            for digit in range(10):
                # Generate multiple samples (batch_size=4) and save the first one
                z = torch.randn(4, self.config.latent_dim).to(self.config.device)
                label = torch.tensor([digit] * 4, device=self.config.device)
                label_onehot = label_to_onehot(label)
                
                gen_imgs = self.generator(z, label_onehot)
                save_image(gen_imgs[0], f'rl_samples/digit_{digit}_epoch_{epoch}.png',
                          normalize=True)
        
        # Switch back to training mode
        self.generator.train()

    def save_models(self):
        """Save the refined models"""
        os.makedirs('refined_models', exist_ok=True)
        torch.save(self.generator.state_dict(), 'refined_models/generator_refined.pth')
        torch.save(self.discriminator.state_dict(), 'refined_models/discriminator_refined.pth')

##################################
# Main Training Loop
##################################
def main():
    config = Config()
    print(f"Using device: {config.device}")

    # Load random samples from collected_dataset
    try:
        real_samples = load_random_samples('collected_dataset', config.num_samples_per_digit)
        print("Loaded random samples from collected_dataset")
    except Exception as e:
        print(f"Error loading samples: {str(e)}")
        return

    # Initialize and train the RL-GAN
    trainer = RLGANTrainer(config)
    trainer.train(real_samples)
    trainer.save_models()
    print("Training complete. Models saved in 'refined_models' directory.")

if __name__ == "__main__":
    main()