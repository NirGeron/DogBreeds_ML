import os
import numpy as np
import cv2
from PIL import Image

def remove_background_grabcut(image_pil):
    """
    Removes the background from an image using OpenCV's GrabCut algorithm.
    """
    image_np = np.array(image_pil)
    
    if image_np.shape[-1] == 4:  # Handle images with alpha channel
        image_np = image_np[:, :, :3]
    
    mask = np.zeros(image_np.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define a rectangle around the main object
    rect = (10, 10, image_np.shape[1] - 10, image_np.shape[0] - 10)

    # Apply GrabCut algorithm
    cv2.grabCut(image_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Convert mask to binary format
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Apply mask to remove background
    result = image_np * mask2[:, :, np.newaxis]

    return Image.fromarray(result)

def process_images(input_folder, output_folder, image_size=(128, 128)):
    """
    Removes background from all images in the input folder and saves them in a different output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create main output directory if not exists

    for class_name in os.listdir(input_folder):
        class_input_dir = os.path.join(input_folder, class_name)
        class_output_dir = os.path.join(output_folder, class_name)

        if not os.path.isdir(class_input_dir):
            continue  # Skip non-directory files

        os.makedirs(class_output_dir, exist_ok=True)  # Create class subdirectory if not exists
        
        for image_file in os.listdir(class_input_dir):
            img_path = os.path.join(class_input_dir, image_file)
            
            try:
                with Image.open(img_path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    # Remove background
                    processed_img = remove_background_grabcut(pil_img)
                    # Resize the image
                    # processed_img = processed_img.resize(image_size, Image.BILINEAR)

                    # Save the processed image
                    save_path = os.path.join(class_output_dir, image_file)
                    processed_img.save(save_path)

            except Exception as e:
                print(f"Skipping file {img_path}: {e}")

# Set paths
input_folder = './dataset'  # Folder containing original images in class subdirectories
output_folder = './removed_bg_dataset'  # Folder to save images after background removal
image_size = (256, 256)

# Run background removal
process_images(input_folder, output_folder, image_size)

print("Background removal completed. Processed images are saved in:", output_folder)
