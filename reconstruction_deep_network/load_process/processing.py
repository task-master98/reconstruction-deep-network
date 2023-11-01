import os
import numpy as np
from PIL import Image
import open3d as o3d

def read_conf_file(conf_file_path):
    image_pairs = []
    intrinsic_params = None
    with open(conf_file_path, 'r') as conf_file:
        lines = conf_file.read().splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("intrinsics_matrix"):
                intrinsic_info = list(map(float, lines[i].split()[1:]))
                fx, fy, cx, cy = intrinsic_info[0], intrinsic_info[4], intrinsic_info[2], intrinsic_info[5]
            elif lines[i].startswith("scan"):
                if intrinsic_params is not None:
                    scan_info = lines[i].split()
                    depth_image, rgb_image, matrix_values = scan_info[1], scan_info[2], list(map(float, scan_info[3:]))
                    image_pairs.append((depth_image, rgb_image, matrix_values, (fx, fy, cx, cy)))
                else:
                    print(f"Skipping scan line without preceding intrinsics: {lines[i]}")
            i += 1
    return image_pairs

def pcd_color_vectorized(depth_image, rgb_image, intrinsic_params, matrix_values):
    height, width = depth_image.shape
    R, T = matrix_values[:3, :3], matrix_values[:3, 3]

    fx, cx, fy, cy = intrinsic_params
    
    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    z = depth_image
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    
    pcd_xyz = np.dstack((x, y, z))
    pcd_xyz = pcd_xyz.reshape(-1, 3)
    pcd_xyz_rgb = np.dot(pcd_xyz - T, np.linalg.inv(R).T)

    j_rgb = ((pcd_xyz_rgb[:, 0] * fx) / pcd_xyz_rgb[:, 2] + cx).astype(int)
    i_rgb = ((pcd_xyz_rgb[:, 1] * fy) / pcd_xyz_rgb[:, 2] + cy).astype(int)

    j_rgb = np.clip(j_rgb, 0, width - 1)
    i_rgb = np.clip(i_rgb, 0, height - 1)

    colors = rgb_image[i_rgb, j_rgb]

    return pcd_xyz, colors

if __name__ == "__main__":
    # Load intrinsic parameters from the conf file
    conf_file_path = "/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_camera_parameters/17DRP5sb8fy.conf"
    image_pairs = read_conf_file(conf_file_path)
    
    rgb_image_path = "/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_color_images"
    depth_image_path = "/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_depth_images"
    
    point_clouds = []
    for depth_image_filename, rgb_image_filename, matrix_values, intrinsics in image_pairs:
        rgb_image = np.array(Image.open(os.path.join(rgb_image_path, rgb_image_filename)))
        depth_image = np.array(Image.open(os.path.join(depth_image_path, depth_image_filename)))
        
        pcd, colors = pcd_color_vectorized(depth_image, rgb_image, intrinsics, np.array(matrix_values).reshape(4, 4))
        
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)

        point_clouds.append(pcd_o3d)
    
    o3d.visualization.draw_geometries(point_clouds)
