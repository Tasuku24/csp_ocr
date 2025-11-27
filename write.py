import tkinter as tk
from PIL import Image, ImageDraw
import os
import random
import time

class DigitDataCollector:
    def __init__(self, master, dataset_path="collected_dataset"):
        self.master = master
        self.master.title("Digit Data Collector")

        # Path where images will be saved
        self.dataset_path = dataset_path
        os.makedirs(self.dataset_path, exist_ok=True)

        # Canvas settings
        self.canvas_width = 280
        self.canvas_height = 280

        # Create a Tkinter canvas with a black background
        self.canvas = tk.Canvas(
            self.master,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black"
        )
        self.canvas.pack()

        # Create a PIL image to match the canvas size
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Bind the mouse drag event for drawing
        self.canvas.bind("<B1-Motion>", self.paint)

        # Prompt label for current digit
        self.prompt_label = tk.Label(self.master, text="", font=("Helvetica", 14))
        self.prompt_label.pack()

        # Buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=5)

        self.save_button = tk.Button(button_frame, text="Save", command=self.save_image)
        self.save_button.pack(side="left", padx=5)

        self.next_button = tk.Button(button_frame, text="Next Digit", command=self.next_digit)
        self.next_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left", padx=5)

        # Initialize a random digit
        self.current_digit = None
        self.next_digit()

    def paint(self, event):
        """
        Draw a white circle (brush stroke) on both the Tkinter canvas and the PIL image.
        """
        radius = 5
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)

        # Draw white on the canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        # Also paint white on the PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def next_digit(self):
        """
        Select a new random digit for the user to draw, clear the canvas, and update the prompt.
        """
        self.clear_canvas()
        self.current_digit = random.randint(0, 9)
        self.prompt_label.config(text=f"Please draw digit {self.current_digit}")

    def save_image(self):
        """
        Save the current PIL image to the dataset folder, under the label subfolder.
        The file name includes a timestamp for uniqueness.
        """
        # Create a label-specific directory
        label_dir = os.path.join(self.dataset_path, str(self.current_digit))
        os.makedirs(label_dir, exist_ok=True)

        # Optionally, downsample to 28×28 if you want smaller images matching MNIST.
        # Remove or comment out this block to save full-size images.
        scaled_img = self.image.resize((28, 28))

        # Construct a file name using a timestamp
        timestamp = int(time.time() * 1000)
        filename = f"image_{timestamp}.png"

        # Save
        filepath = os.path.join(label_dir, filename)
        scaled_img.save(filepath)

        # Notify the user
        self.prompt_label.config(text=f"Saved digit {self.current_digit} to {filepath}")

    def clear_canvas(self):
        """
        Clear the canvas and reset the PIL image to black.
        """
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDataCollector(root, dataset_path="collected_dataset")
    root.mainloop()