import os
import random
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F

def find_bounding_box(img_np):
  """
  Finds the bounding box of non-zero pixels in a (H,W) numpy array.
  Returns (minY, maxY, minX, maxX).
  If no non-zero pixels are found, returns None.
  """
  rows = np.any(img_np, axis=1)
  cols = np.any(img_np, axis=0)
  if not np.any(rows) or not np.any(cols):
    return None
  min_y, max_y = np.where(rows)[0][[0, -1]]
  min_x, max_x = np.where(cols)[0][[0, -1]]
  return min_y, max_y, min_x, max_x

def random_safe_affine(img, max_tries=10):
  """
  Applies random rotation, scaling, and translation while ensuring
  the digit remains within the 28x28 boundary.
  Retries up to `max_tries` times if it doesn't fit.
  """
  width, height = img.size

  for _ in range(max_tries):
    # Random parameters for rotation, scale, and translation
    degrees = random.uniform(-20, 20)
    scale = random.uniform(0.8, 1.2)
    max_translate = 4  # up to 4px in each direction
    translate_x = random.uniform(-max_translate, max_translate)
    translate_y = random.uniform(-max_translate, max_translate)
    shear = random.uniform(-10, 10)

    # Apply the random affine
    transformed = F.affine(
      img,
      angle=degrees,
      translate=(translate_x, translate_y),
      scale=scale,
      shear=shear,
      fill=0  # fill with black
    )

    # Check bounding box of non-zero pixels
    np_img = np.array(transformed)
    bbox = find_bounding_box(np_img)
    if bbox is None:
      # no content, technically 'fits'
      return transformed
    min_y, max_y, min_x, max_x = bbox

    # Ensure bounding box is fully within the image
    if 0 <= min_y and max_y < height and 0 <= min_x and max_x < width:
      return transformed

  # If no valid transform found, return original
  return img

def process_dataset(dataset_path, output_path, limit=None):
  """
  Scans a folder (dataset_path) containing subfolders for each digit.
  Applies random safe affine transform and saves to output_path.
  If limit is specified, only processes up to 'limit' images per digit.
  """
  os.makedirs(output_path, exist_ok=True)
  for digit in sorted(os.listdir(dataset_path)):
    digit_dir = os.path.join(dataset_path, digit)
    if not os.path.isdir(digit_dir):
      continue
    out_dir = os.path.join(output_path, digit)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(digit_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if limit:
      files = files[:limit]

    for fname in files:
      file_path = os.path.join(digit_dir, fname)
      img = Image.open(file_path).convert("L")

      # Resize to 28x28 if not already, to unify
      if img.size != (28, 28):
        img = img.resize((28, 28))

      transformed = random_safe_affine(img)
      out_name = os.path.splitext(fname)[0] + "_tf.png"
      transformed.save(os.path.join(out_dir, out_name))

def main():
  # Example usage:
  mnist_input = "mnist_data/MNIST/raw"  # or wherever your MNIST is stored in subfolders
  generated_input = "generated_dataset"           # your generated set

  mnist_output = "transformed_mnist"
  generated_output = "transformed_generated"

  # Process both sets, optionally limit images per digit if desired
  process_dataset(mnist_input, mnist_output, limit=None)
  process_dataset(generated_input, generated_output, limit=None)

if __name__ == "__main__":
  main()