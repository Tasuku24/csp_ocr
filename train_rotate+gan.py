import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

batch_size = 1024
learning_rate = 0.001
epochs = 30

if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(f"Using device: {device}")

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((28, 28)),                  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

aug_transform = transforms.Compose([
  transforms.RandomRotation(30),          
  transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  
  transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
  transforms.Grayscale(num_output_channels=1),
  transforms.Resize((28, 28)),
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = datasets.MNIST(
    root='mnist_data',
    train=True,
    transform=aug_transform,
    download=True
)

generated_dataset_path = "generated_dataset"
if not os.path.exists(generated_dataset_path):
    print("Warning: generated_dataset folder not found. Only MNIST data will be used.")
    combined_dataset = mnist_dataset
else:
    generated_dataset = datasets.ImageFolder(
        root=generated_dataset_path,
        transform=data_transform
    )
    # Combine both MNIST (augmented) and generated dataset
    combined_dataset = ConcatDataset([mnist_dataset, generated_dataset])

##################################
# Create Dataloader
##################################
train_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True)

# For validation, load the standard MNIST test set with typical transform
test_dataset = datasets.MNIST(
    root='mnist_data',
    train=False,
    transform=data_transform,  # no augmentation on test set
    download=True
)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

##################################
# Model Definition: Improved CNN
##################################
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*3*3, 128)
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

        x = x.view(-1, 64*3*3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

##################################
# Initialize Model, Loss, Optimizer
##################################
model = ImprovedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

##################################
# Training Loop
##################################
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

##################################
# Validation / Testing
##################################
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

##################################
# Prompt for Model Name & Save
##################################
model_name = input("Enter a name for the saved model (e.g., 'my_digit_model'): ")
if not model_name.strip():
  model_name = "default_mnist_model"

# Ensure the directory exists
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

filename = os.path.join(save_dir, f"{model_name}.pth")
torch.save(model.state_dict(), filename)
print(f"Model weights saved to '{filename}'")