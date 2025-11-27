import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

############################################
# 1. Model Architecture Must Match Training
############################################
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

############################################
# 2. Load All Models from Folder
############################################
def load_all_models(folder_path, device):
    """
    Scans the folder for *.pth files and loads each into an ImprovedCNN instance.
    Returns a dict {model_name: model_object}.
    """
    models_dict = {}
    pth_files = glob.glob(os.path.join(folder_path, "*.pth"))
    
    if not pth_files:
        print(f"No .pth files found in {folder_path}. Skipping load.")
        return models_dict

    for file_path in pth_files:
        # Model name = filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        model = ImprovedCNN()
        # If using PyTorch 2.1+, you can add "weights_only=True" to torch.load(...) below:
        # state_dict = torch.load(file_path, map_location=device, weights_only=True)
        state_dict = torch.load(file_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        models_dict[base_name] = model

    return models_dict

############################################
# 3. Main Benchmark Routine
############################################
def main():
    # Path to your folder of saved .pth model files
    models_folder = "saved_models"

    # Path to your collected dataset root folder
    dataset_path = "collected_dataset"

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3A. Load all models
    models_dict = load_all_models(models_folder, device)
    if not models_dict:
        return  # No models loaded, exit

    # 3B. Prepare your dataset
    #    We assume a folder structure like collected_dataset/0, collected_dataset/1, etc.
    if not os.path.exists(dataset_path):
        print(f"Error: The dataset folder '{dataset_path}' does not exist.")
        return

    # 3C. Define Data Transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensures single-channel if needed
        transforms.Resize((28, 28)),                  # Matches MNIST dimension
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3D. Create Dataset & DataLoader
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    # On Windows, set num_workers=0 to avoid spawn issues. Increase if desired on other OSes.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # 3E. Evaluate Each Model
    for model_name, model in models_dict.items():
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(f"[{model_name}] Accuracy on collected dataset: {accuracy:.2f}%")

############################################
# 4. Windows-Safe Entry Point
############################################
if __name__ == "__main__":
    main()