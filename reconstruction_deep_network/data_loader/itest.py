"""
This file will be used to extract, load and transform the Matterport dataset
"""
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image

from reconstruction_deep_network.preprocessing import depth_to_pcd_process

from reconstruction_deep_network.utils import voxel
from reconstruction_deep_network.utils import visualization




import os
import configparser
import reconstruction_deep_network

package_dir = reconstruction_deep_network.__path__[0]
data_dir = os.path.join(os.path.dirname(package_dir), "data")

class MatterPortData:
    matterport_parent_dir = os.path.join(data_dir, "v1", "scans")
    matterport_dir_names = {
        "cameras": "cameras",
        "house_segments": "house_segmentations",
        "image_overlap": "image_overlap",
        "camera_intrinsics": "matterport_camera_intrinsics",
        "camera_poses": "matterport_camera_poses",
        "color_images": "matterport_color_images",
        "depth_images": "matterport_depth_images",
        "undist_depth": "undistorted_depth_images",
        "undist_color": "undistorted_color_images",
        "undist_camera_params": "undistorted_camera_parameters",
        "region_segments": "matterport_region_segmentations", 
    }
    n_scenes = 6
    camera_indices = 3

    def __init__(self, scan_hash: str):
        self.scan_hash = scan_hash
        self.dataset_version = os.path.join(self.matterport_parent_dir, scan_hash)
        self.config_dict, self.intrinsic_dict, self.extrinsic_dict = self.intialize_undist_camera_params()
    
    def load_color_image(self, image_name: str, cam_index: int, scene_index: int):
        self._file_path_validator(cam_index, scene_index)
        file_type = self.matterport_dir_names["color_images"]
        img_name = f"{image_name}_i{cam_index}_{scene_index}.jpg"
        file_path = os.path.join(self.dataset_version, file_type, img_name)
        img_arr = np.array(Image.open(file_path))
        return img_arr
    
    def load_depth_image(self, image_name: str, cam_index: int, scene_index: int):
        self._file_path_validator(cam_index, scene_index)
        file_type = self.matterport_dir_names["depth_images"]
        img_name = f"{image_name}_d{cam_index}_{scene_index}.png"
        file_path = os.path.join(self.dataset_version, file_type, img_name)
        img_arr = np.array(Image.open(file_path))
        return img_arr
    
    def load_color_depth_pair(self, image_name: str, cam_index: int, scene_index: int):
        return self.load_color_image(image_name, cam_index, scene_index), \
               self.load_depth_image(image_name, cam_index, scene_index)
    
    def load_camera_intrinsics(self, image_name: str, cam_index: int):
        assert cam_index in range(self.camera_indices)
        file_type = self.matterport_dir_names["camera_intrinsics"]
        file_name = f"{image_name}_intrinsics_{cam_index}.txt"
        file_path = os.path.join(self.dataset_version, file_type, file_name)
        parsed_content = self._parse_txt_file(file_path)
        return parsed_content
    
    def load_camera_extrinsics(self, image_name: str, cam_index: int, scene_index: int):
        self._file_path_validator(cam_index, scene_index)
        file_type = self.matterport_dir_names["camera_poses"]
        file_name = f"{image_name}_pose_{cam_index}_{scene_index}.txt"
        file_path = os.path.join(self.dataset_version, file_type, file_name)
        parsed_content = self._parse_txt_file(file_path)
        camera_extrinsics = parsed_content.reshape((4, 4))
        rotation_mat, translation_vec = camera_extrinsics[:3, :3], camera_extrinsics[:-1, 3]
        return rotation_mat, translation_vec
    
    def load_undistorted_color_depth_pair(self, image_name: str, cam_index: int, scene_index: int):
        return self.load_undistorted_color_image(image_name, cam_index, scene_index), \
               self.load_undistorted_depth_image(image_name, cam_index, scene_index)
    
    def load_undistorted_color_image(self, image_name: str, cam_index: int, scene_index: int):
        self._file_path_validator(cam_index, scene_index)
        file_type = self.matterport_dir_names["undist_color"]
        img_name = f"{image_name}_i{cam_index}_{scene_index}.jpg"
        file_path = os.path.join(self.dataset_version, file_type, img_name)
        print(img_name)
        img_arr = np.array(Image.open(file_path))
        return img_arr
    
    def load_undistorted_depth_image(self, image_name: str, cam_index: int, scene_index: int):
        self._file_path_validator(cam_index, scene_index)
        file_type = self.matterport_dir_names["undist_depth"]
        img_name = f"{image_name}_d{cam_index}_{scene_index}.png"
        file_path = os.path.join(self.dataset_version, file_type, img_name)
        print(img_name)
        img_arr = np.array(Image.open(file_path))
        return img_arr
    
    def load_undist_camera_params(self, image_name, cam_index: int, scene_index: int):
        self._file_path_validator(cam_index, scene_index)
        intrinsic_matrix = self.intrinsic_dict[(image_name, cam_index)]
        extrinsic_matrix = np.array(self.extrinsic_dict[(image_name, cam_index, scene_index)]).reshape((4, 4))
        R, T = extrinsic_matrix[:3, :3], extrinsic_matrix[:-1, 3]
        return intrinsic_matrix, R, T
    
    
    def intialize_undist_camera_params(self):
        file_type = self.matterport_dir_names["undist_camera_params"]
        file_name = f"{self.scan_hash}.conf"
        file_path = os.path.join(self.dataset_version, file_type, file_name)
        
        # Create dictionaries to store configuration and metadata
        config_dict = {}
        metadata_dict = {}

        is_intrinsic = True
        # Open and read the .conf file
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        
        clean_lines = list(filter(lambda x: x != "", lines))
        
        
        # iterate over first 4 lines
        for content in clean_lines[:4]:
            key, val = content.split()
            if val.isdigit():
                config_dict[key] = int(val)
            else:
                config_dict[key] = val
        
        scan_metadata = clean_lines[4:]
        group_size = 7
        spliced_groups = [scan_metadata[i:i+group_size] for i in range(0, len(scan_metadata), group_size)]

        intrinsic_mat_metadata = {}
        for group in spliced_groups:
            intrinsic_mat_data, first_scan_line = group[0], group[1]
            intrinsic_mat_row = intrinsic_mat_data.split()
            scan_row = first_scan_line.split()[1]
            panorama_uid, cam_idx, yaw_idx = scan_row.split(".")[0].split("_")
            intrinsic_mat_metadata[(panorama_uid, int(cam_idx[1]))] = [float(num) for num in intrinsic_mat_row[1:]]

        extrinsic_mat_metadata = {}
        for group in spliced_groups:
            for extrinsic_row in group[1:]:
                extrinsic_items = extrinsic_row.split()
                extrinsic_items.remove("scan")
                img_name = extrinsic_items[0].split(".")[0]
                panorama_uid, cam_idx, yaw_idx = img_name.split("_")
                extrinsic_mat_metadata[(panorama_uid, int(cam_idx[1]), int(yaw_idx))] = [float(num) for num in extrinsic_items[2:]]


        return config_dict, intrinsic_mat_metadata, extrinsic_mat_metadata


    def _parse_txt_file(self, file_path: str):

        with open(file_path, 'r') as f:
            line = f.read()

            if line:
                numbers_as_str = line.split("\n")
                numbers_as_str.remove("")
                if len(numbers_as_str) > 1:
                    numbers = []
                    for row in numbers_as_str:
                        row_numbers = row.split()
                        numbers.extend([float(num) for num in row_numbers])
                elif len(numbers_as_str) == 1:
                    numbers_as_str = numbers_as_str[0].split()
                    numbers = [float(num) for num in numbers_as_str]
            else:
                raise ValueError("File is empty")
        
        num_arr = np.array(numbers)
        return num_arr

    def _file_path_validator(self, camera_index: int, scene_index: int):
        assert camera_index in range(self.camera_indices)
        assert scene_index in range(self.n_scenes)




if __name__ == "__main__":

    from reconstruction_deep_network.preprocessing import depth_to_pcd_process
    from reconstruction_deep_network.utils import voxel
    from reconstruction_deep_network.utils import visualization

    scan_hash = "17DRP5sb8fy"
    data_loader = MatterPortData(scan_hash)
    # "00ebbf3782c64d74aaf7dd39cd561175_i2_0.jpg"
    image_name = "0f37bd0737e349de9d536263a4bdd60d"
    cam_indices = 2
    scene_index = 0
    # "00ebbf3782c64d74aaf7dd39cd561175_i2_5.jpg"

    pcd = []

    for cam_index in range(2, 3):
        for yaw_idx in range(4, 6):
            print(cam_index, yaw_idx)
            _, depth_image = data_loader.load_undistorted_color_depth_pair(image_name, cam_index, yaw_idx)
            # depth_image = depth_to_pcd_process.convert_image_to_grayscale(depth_image)
            camera_intrinsics, R, T = data_loader.load_undist_camera_params(image_name, cam_index, yaw_idx)

            point_cloud = depth_to_pcd_process.pcd_vectorized(depth_image, camera_intrinsics, (R, T))

            pcd.extend(point_cloud.tolist())
    
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    
    # Visualize:
    pcd_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    voxel.voxelize_and_display(pcd_o3d, 1)
    visualization.visualize_point_cloud(pcd_o3d)

    data_loader.intialize_undist_camera_params()
   


    
