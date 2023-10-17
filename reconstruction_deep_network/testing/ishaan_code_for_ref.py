import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from load_dataset import NYUv2Dataset
import os

#  Depth Camera params
FX_DEPTH = 5.8262448167737955e+02
FY_DEPTH = 5.8269103270988637e+02
CX_DEPTH = 3.1304475870804731e+02
CY_DEPTH = 2.3844389626620386e+02

# RGB Camera params
FX_RGB = 5.1885790117450188e+02
FY_RGB = 5.1946961112127485e+02
CX_RGB = 3.2558244941119034e+02
CY_RGB = 2.5373616633400465e+02

# Rotation Matrix
R = -np.array([[9.9997798940829263e-01, 5.0518419386157446e-03, 4.3011152014118693e-03],
                [-5.0359919480810989e-03, 9.9998051861143999e-01, -3.6879781309514218e-03],
                [-4.3196624923060242e-03, 3.6662365748484798e-03, 9.9998394948385538e-01]])

# Translation
T = np.array([2.5031875059141302e-02, -2.9342312935846411e-04, 6.6238747008330102e-04])

_visualizer_params = {
    "zoom": 0.3412,
    "front": [0.4257, -0.2125, -0.8795],
    "lookat": [2.6172, 2.0475, 1.532],
    "up": [-0.0694, -0.9768, 0.2024]
}

def convert_image_to_grayscale(depth_image):
    depth_grayscale = np.array(256 * (depth_image / 0x0fff), dtype=np.uint8)
    return depth_grayscale

def pcd_nested_loops(depth_image: np.ndarray):
    """
    Converts depth image to point cloud
    """
    height, width = depth_image.shape
    pcd = []
    for i in range(height):
        for j in range(width):
            z = depth_image[i][j]
            x = (j - CX_DEPTH) * depth_image[i][j] / FX_DEPTH
            y = (i - CY_DEPTH) * depth_image[i][j] / FY_DEPTH
            pcd.append([x, y, z])
    
    return np.array(pcd)

def pcd_vectorized(depth_image: np.ndarray):
    height, width = depth_image.shape
    
    # Create arrays for i and j values using meshgrid
    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Compute x, y, and z using vectorized operations
    z = depth_image
    x = (j - CX_DEPTH) * z / FX_DEPTH
    y = (i - CY_DEPTH) * z / FY_DEPTH
    
    # Stack x, y, and z to form the point cloud
    pcd = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    
    return pcd

def pcd_color(depth_image: np.ndarray, rgb_image: np.ndarray):
    height, width = depth_image.shape
    colors = []
    pcd = []
    for i in range(height):
        for j in range(width):
            
            z = depth_image[i][j]
            x = (j - CX_DEPTH) * z / FX_DEPTH
            y = (i - CY_DEPTH) * z / FY_DEPTH

            [x_RGB, y_RGB, z_RGB] = np.linalg.inv(R).dot([x, y, z]) - np.linalg.inv(R).dot(T)

            j_rgb = int((x_RGB * FX_RGB) / z_RGB + CX_RGB + width / 2)
            i_rgb = int((y_RGB * FY_RGB) / z_RGB + CY_RGB)

            pcd.append([x, y, z])

            # Add the color of the pixel if it exists:
            
            if 0 <= j_rgb < width and 0 <= i_rgb < height:
                colors.append(rgb_image[i_rgb][j_rgb])
            else:
                colors.append([0., 0., 0.])
    
    return [np.array(pcd), np.array(colors)]

def pcd_color_vectorized(depth_image: np.ndarray, rgb_image: np.ndarray):
    height, width = depth_image.shape

    # Vectorized point cloud computation
    pcd_xyz = pcd_vectorized(depth_image)
    
    # Convert x, y, and z to RGB coordinates
    pcd_xyz_rgb = np.dot(pcd_xyz - T, np.linalg.inv(R).T)
    
    # Calculate the corresponding RGB indices
    j_rgb = ((pcd_xyz_rgb[:, 0] * FX_RGB) / pcd_xyz_rgb[:, 2] + CX_RGB + width / 2).astype(int)
    i_rgb = ((pcd_xyz_rgb[:, 1] * FY_RGB) / pcd_xyz_rgb[:, 2] + CY_RGB).astype(int)
    
    # Clip indices to ensure they are within image boundaries
    j_rgb = np.clip(j_rgb, 0, width - 1)
    i_rgb = np.clip(i_rgb, 0, height - 1)
    
    # Extract colors from the RGB image based on calculated indices
    colors = rgb_image[i_rgb, j_rgb]
    
    return [pcd_xyz, colors]


if __name__ == "__main__":

    nyu2_train = NYUv2Dataset("train")
    rgb_image, depth = nyu2_train.extract_example_by_category("living_room_0038_out")
    
    depth_image = depth.astype(np.uint16)
    
    pcd, colors = pcd_color_vectorized(depth_image, rgb_image)

    
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(colors / 255))
    # Visualize:
    pcd_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_o3d])

            
    