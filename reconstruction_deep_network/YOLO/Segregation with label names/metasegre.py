import os
import shutil

# Constants
metadata_file_path = '/Users/mario/Desktop/Project/reconstruction-deep-network/reconstruction_deep_network/YOLO/metadata.txt'
source_image_folder = '/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/testing3/object'
destination_root_folder = '/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/testing3/meta_segre'

# Ensure destination root folder exists
if not os.path.exists(destination_root_folder):
    os.makedirs(destination_root_folder)

# Parse metadata file and organize images
with open(metadata_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        filename, label, *_ = line.strip().split(',')
        
        # Create folder for the label if it doesn't exist
        label_folder = os.path.join(destination_root_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        # Copy the image to the label's folder
        source_path = os.path.join(source_image_folder, filename)
        dest_path = os.path.join(label_folder, filename)
        
        shutil.copy(source_path, dest_path)

print("Images organized by labels!")
