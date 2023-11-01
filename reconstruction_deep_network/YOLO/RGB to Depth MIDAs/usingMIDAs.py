import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()

# Function to estimate depth
def estimate_depth(img_path):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(384),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_img = transform(img).unsqueeze(0)

    # Depth estimation
    with torch.no_grad():
        depth = midas(input_img)

    depth = depth.squeeze().cpu().numpy()
    return depth

def process_directory(root_dir):
    # Loop through all sub-directories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Check if file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, file)
                
                # Convert RGB image to depth image
                depth_map = estimate_depth(img_path)

                # Convert depth map to 8-bit grayscale for visualization
                depth_map_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                
                # Save depth image
                output_path = os.path.join(subdir, "depth_" + file)
                cv2.imwrite(output_path, depth_map_normalized)
                print(f"Processed {img_path} -> {output_path}")

root_dir = "/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/testing3/meta_segre"
process_directory(root_dir)
