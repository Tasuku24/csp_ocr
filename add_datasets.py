import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

##################################
# cGAN Generator Definition
# Must match the architecture used during training.
##################################
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

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
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels_onehot):
        x = torch.cat([noise, labels_onehot], dim=1)
        out = self.model(x)
        out = out.view(out.size(0), 1, 28, 28)
        return out

##################################
# Helper: One-Hot Encode Labels
##################################
def label_to_onehot(labels, num_classes=10):
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot[range(batch_size), labels] = 1
    return one_hot

##################################
# Load Generator
##################################
def load_generator(
    # model_path="cgan_models/generator.pth", 
    model_path="refined_models/generator_refined.pth",
    latent_dim=100, 
    num_classes=10, 
    device="cpu"
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Generator file not found at: {model_path}")
    gen = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    return gen

##################################
# Main Function: Add N Datasets
##################################
def add_datasets(generator, n, device="cpu", output_dir="generated_dataset2", batch_size=64):
    """
    For each digit [0..9], generate n new images using the provided generator.
    Save them in output_dir/<digit>/, continuing the numbering of existing images.
    """

    latent_dim = generator.latent_dim
    num_classes = generator.num_classes

    # Ensure the digit subfolders exist
    os.makedirs(output_dir, exist_ok=True)
    for d in range(num_classes):
        digit_dir = os.path.join(output_dir, str(d))
        os.makedirs(digit_dir, exist_ok=True)

    # We'll do it digit by digit. For each digit:
    for digit in range(num_classes):
        # Find how many images are currently in the folder to continue numbering
        digit_dir = os.path.join(output_dir, str(digit))
        existing_images = [
            f for f in os.listdir(digit_dir) if f.endswith(".png") or f.endswith(".jpg")
        ]
        if not existing_images:
            start_index = 0
        else:
            # Parse out the highest numeric file name prefix if possible
            indices = []
            for img_name in existing_images:
                name_part = os.path.splitext(img_name)[0]  # e.g. "45" from "45.png"
                try:
                    idx = int(name_part)
                    indices.append(idx)
                except ValueError:
                    pass
            start_index = max(indices) + 1 if indices else 0

        import torchvision.transforms as transforms

        # We want to generate n images total for this digit
        count = 0
        while count < n:
            z = torch.randn(batch_size, latent_dim, device=device)
            labels_int = torch.full((batch_size,), digit, dtype=torch.long, device=device)
            labels_oh = label_to_onehot(labels_int, num_classes=num_classes)
            with torch.no_grad():
              gen_imgs = generator(z, labels_oh)

            for img in gen_imgs:
              if count >= n:
                break
              file_index = start_index + count
              save_path = os.path.join(digit_dir, f"{file_index}.png")
              pil_img = transforms.ToPILImage()(img.cpu()).convert("L")
              pil_img.save(save_path)
              count += 1

        print(f"Digit {digit}: created {n} new images (from index {start_index} to {start_index + n - 1}).")

def main():
    if torch.backends.mps.is_available():
      device = torch.device("mps")
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model automatically from cgan_models folder
    # model_path = "cgan_models/generator.pth"
    model_path="refined_models/generator_refined.pth"
    try:
        generator = load_generator(model_path=model_path, device=device)
    except FileNotFoundError as e:
        print(e)
        return

    # Prompt user for number n
    try:
        n = int(input("Enter the number of datasets to add for each digit: "))
    except ValueError:
        print("Invalid input. Exiting.")
        return

    add_datasets(generator, n, device=device, output_dir="generated_dataset", batch_size=64)
    print("Done adding new datasets.")

if __name__ == "__main__":
    main()