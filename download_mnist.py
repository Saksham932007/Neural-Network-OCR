import requests
import gzip
import numpy as np
import pandas as pd
import os

# --- Configuration ---
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}
# UPDATED: The original yann.lecun.com URL is often down. Using a stable TensorFlow mirror.
BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
OUTPUT_DIR = "." # Output to the current directory

# --- Helper Functions ---
def download_file(url, local_filename):
    """Downloads a file from a URL and saves it locally."""
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists. Skipping download.")
        return
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def load_mnist_images(filename):
    """Loads MNIST images from the gzipped ubyte file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with gzip.open(filepath, 'rb') as f:
        # The first 16 bytes are magic number, number of images, rows, and columns
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    # Images are 28x28 pixels
    return data.reshape(-1, 28 * 28)

def load_mnist_labels(filename):
    """Loads MNIST labels from the gzipped ubyte file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with gzip.open(filepath, 'rb') as f:
        # The first 8 bytes are magic number and number of items
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

def save_to_csv(images, labels, filename):
    """Saves images and labels to a CSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        print(f"{filepath} already exists. Skipping creation.")
        return
        
    print(f"Creating {filepath}...")
    # Create a DataFrame where the first column is the label and the rest are pixel values
    df = pd.DataFrame(images)
    df.insert(0, 'label', labels)
    df.to_csv(filepath, index=False)
    print(f"Successfully saved to {filepath}")

# --- Main Execution ---
def main():
    """Main function to download and process the MNIST dataset."""
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download all necessary files
    for key, filename in FILES.items():
        download_file(BASE_URL + filename, os.path.join(OUTPUT_DIR, filename))

    # Load the data
    print("\nLoading data from files...")
    train_images = load_mnist_images(FILES["train_images"])
    train_labels = load_mnist_labels(FILES["train_labels"])
    test_images = load_mnist_images(FILES["test_images"])
    test_labels = load_mnist_labels(FILES["test_labels"])

    # Save to CSV
    save_to_csv(train_images, train_labels, 'mnist_train.csv')
    save_to_csv(test_images, test_labels, 'mnist_test.csv')
    
    print("\nAll done!")

if __name__ == "__main__":
    main()

