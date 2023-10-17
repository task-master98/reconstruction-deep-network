import numpy as np
from PIL import Image
import open3d as o3d
import os

def read_conf_file(conf_file_path):
    intrinsic_params = []
    image_pairs = []
    with open(conf_file_path, 'r') as conf_file:
        lines = conf_file.read().splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("intrinsics_matrix"):
                intrinsics = list(map(float, lines[i].split()[1:]))
                intrinsic_params.append(intrinsics)
            elif lines[i].startswith("scan"):
                depth_image, rgb_image, matrix_values = lines[i].split()[1], lines[i].split()[2], list(map(float, lines[i].split()[3:]))
                image_pairs.append((depth_image, rgb_image, matrix_values, intrinsics))
            i += 1
    return image_pairs

def pcd_vectorized(depth_image, intrinsic_params):
    height, width = depth_image.shape
    pcd_list = []


    
    fx, fy, cx, cy = intrinsic_params

    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    z = depth_image
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy

    pcd = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    pcd_list.append(pcd)

    return pcd_list

def pcd_color_vectorized(depth_image, rgb_image, intrinsic_params, matrix_values):
    height, width = depth_image.shape
    R, T = matrix_values[:3, :3], matrix_values[:-1, 3]

    if len(intrinsic_params) < 4:
        raise ValueError("Intrinsic parameters must have at least 4 values.")

    pcd_list = []

    # for intrinsic in intrinsic_params:
    fx = intrinsic_params[0]
    cx = intrinsic_params[2]
    fy = intrinsic_params[4]
    cy = intrinsic_params[5]
    
    pcd_xyz = np.array(pcd_vectorized(depth_image, [fx, cx, fy, cy]))
    pcd_xyz_rgb = np.dot(pcd_xyz - T, np.linalg.inv(R).T)
    

    j_rgb = ((pcd_xyz_rgb[:, 0] * fx) / pcd_xyz_rgb[:, 2] + cx).astype(int)
    i_rgb = ((pcd_xyz_rgb[:, 1] * fy) / pcd_xyz_rgb[:, 2] + cy).astype(int)

    j_rgb = np.clip(j_rgb, 0, width - 1)
    i_rgb = np.clip(i_rgb, 0, height - 1)

    colors = rgb_image[i_rgb, j_rgb]

    pcd_list.append([pcd_xyz, colors])

    return pcd_xyz, colors


if __name__ == "__main__":
    conf_file_path = "/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_camera_parameters/17DRP5sb8fy.conf"  # Update with your conf file path
    image_pairs = read_conf_file(conf_file_path)
    
    rgb_image_path = "/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_color_images"  # Update with your RGB images path
    depth_image_path = "/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/17DRP5sb8fy/undistorted_depth_images"  # Update with your depth images path
    
    point_clouds = []
    for depth_image_filename, rgb_image_filename, matrix_values, intrinsics in image_pairs[12:13]:
        rgb_image = np.array(Image.open(os.path.join(rgb_image_path, rgb_image_filename)))
        depth_image = np.array(Image.open(os.path.join(depth_image_path, depth_image_filename)))
        print(depth_image_filename)
        
        pcd, colors = pcd_color_vectorized(depth_image, rgb_image, intrinsics, np.array(matrix_values).reshape(4, 4))
        
        pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd.squeeze())  # set pcd_np as the point cloud points

        pcd_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        o3d.visualization.draw_geometries([pcd_o3d])