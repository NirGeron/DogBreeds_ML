import zipfile
import os


def extract_zip(zip_path, extract_to):
    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    # Open and extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extraction completed. Files are extracted to: {extract_to}")


    
zip_file_path = "/notebooks/dog_breeds_dataset.zip"
output_directory = "dataset"


extract_zip(zip_file_path, output_directory)
