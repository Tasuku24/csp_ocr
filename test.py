import os
import glob
import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

##################################
# 1. CNN Architecture
##################################
# Must match exactly how you defined and trained your models
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
# 2. Load All Models from a Folder
##################################
def load_all_models_from_folder(folder_path, device):
    """
    Loads all .pth files in the specified folder using the same CNN architecture.
    Returns a dictionary: {model_filename_no_ext: model_instance}.
    """
    models_dict = {}
    # Find all .pth files
    pth_files = glob.glob(os.path.join(folder_path, "*.pth"))
    
    if not pth_files:
        print(f"No model files found in {folder_path}.")
        return models_dict

    for pth_file in pth_files:
        model_name = os.path.splitext(os.path.basename(pth_file))[0]
        # Create model instance and load weights
        model = ImprovedCNN()
        # If you're using PyTorch 2.1+, you can do weights_only=True:
        # state_dict = torch.load(pth_file, map_location=device, weights_only=True)
        state_dict = torch.load(pth_file, map_location=device)  # For older versions
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        models_dict[model_name] = model
    
    return models_dict

##################################
# 3. GUI Class: Use a Larger Canvas, Then Downsample
##################################
class DigitGUIAllModels:
    def __init__(self, master, models_dict, device):
        self.master = master
        self.master.title("Compare Predictions from Multiple Models")

        self.models_dict = models_dict
        self.device = device

        # Canvas settings
        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()

        # PIL image to capture drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Bind the mouse event
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack()

        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left", padx=5)

        # Label for predictions
        self.result_label = tk.Label(self.master, text="Draw a digit and click Predict")
        self.result_label.pack(pady=5)

        # Transform that we apply before feeding to the models
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def paint(self, event):
        """
        Draw white circles on a black background for the digit.
        """
        radius = 5
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def predict_digit(self):
        """
        Downsample to 28x28, run inference with all saved models, and display the predictions.
        """
        scaled_img = self.image.resize((28, 28))
        img_tensor = self.transform(scaled_img).unsqueeze(0).to(self.device)

        predictions_text = "Predictions:\n"
        for model_name, model in self.models_dict.items():
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, dim=1)
            predictions_text += f"{model_name}: {predicted.item()}\n"

        self.result_label.config(text=predictions_text)

    def clear_canvas(self):
        """
        Clear the canvas and PIL image.
        """
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.result_label.config(text="Draw a digit and click Predict")

##################################
# 4. Main Section to Run Everything
##################################
if __name__ == "__main__":
    # Folder where your model .pth files are located
    models_folder = "saved_models"

    # Determine device
    if torch.backends.mps.is_available():
      device = torch.device("mps")
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all models in the folder
    models_dict = load_all_models_from_folder(models_folder, device)

    # Start the Tkinter GUI
    root = tk.Tk()
    app = DigitGUIAllModels(root, models_dict, device)
    root.mainloop()