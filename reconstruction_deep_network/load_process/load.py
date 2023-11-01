import os
from PIL import Image
import numpy as np

# Directory paths
color_image_dir = '/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_color_images'
depth_image_dir = '/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_depth_images'

# List files in the directories
color_image_files = os.listdir(color_image_dir)
depth_image_files = os.listdir(depth_image_dir)

# Filter out non-image files (e.g., Thumbs.db)
color_image_files = [f for f in color_image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
depth_image_files = [f for f in depth_image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Load color and depth images
for color_image_file, depth_image_file in zip(color_image_files, depth_image_files):
    color_image = np.array(Image.open(os.path.join(color_image_dir, color_image_file)))
    depth_image = np.array(Image.open(os.path.join(depth_image_dir, depth_image_file)))

    # Process and use the images for your specific task
