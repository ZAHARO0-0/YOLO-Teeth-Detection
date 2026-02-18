# Standard YOLO image detection script with file upload dialog
# Date: 2026-02-18
# Description: Detect teeth conditions in a user-selected image using YOLO model.

import os
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------------------
# CONFIGURATION
# -------------------------------

# YOLO model weights file (placed in the same folder as this script)
MODEL_FILE = "best.pt"  # just the filename, no folder needed

# -------------------------------
# MAIN SCRIPT
# -------------------------------

def main():
    """
    Main function to run YOLO detection on a user-selected image.
    """
    # Hide main tkinter window
    Tk().withdraw()

    # Ask user to select an image file
    img_path = askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not img_path:
        print("No file selected. Exiting.")
        return

    # Load the YOLO model (from same folder as script)
    if not os.path.isfile(MODEL_FILE):
        print(f"Model file '{MODEL_FILE}' not found in the project folder.")
        print("Please download or place your YOLO model weights in the same folder as this script.")
        return

    model = YOLO(MODEL_FILE)

    # Run detection on the selected image
    results = model(img_path)

    # Display results
    for r in results:
        print(f"File: {os.path.basename(img_path)}")
        print(f"→ Accuracy: {r.boxes.conf.mean().item() * 100:.1f}%")
        # Show the annotated image with bounding boxes
        r.show()

if __name__ == "__main__":
    main()
