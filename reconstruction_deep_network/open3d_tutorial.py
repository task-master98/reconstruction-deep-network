import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import copy
import os
import sys
import json

_visualizer_params = {
    "zoom": 0.3412,
    "front": [0.4257, -0.2125, -0.8795],
    "lookat": [2.6172, 2.0475, 1.532],
    "up": [-0.0694, -0.9768, 0.2024]
}

def voxelize_and_display(pcd, size, display=True):
    downpcd = pcd.voxel_down_sample(voxel_size = size)
    print(downpcd)
    if display:
        o3d.visualization.draw_geometries([downpcd],
                                        **_visualizer_params)
    
    return downpcd

def estimate_normals_and_display(pcd, display=True):
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if display:
        o3d.visualization.draw_geometries([pcd],
                                        **_visualizer_params,
                                        point_show_normal=True)

def crop_point_cloud(pcd, display = True):
    crop_data = o3d.data.DemoCropPointCloud()
    vol = o3d.visualization.read_selection_polygon_volume(crop_data.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)
    
    with open(crop_data.cropped_json_path) as f:
        data = json.load(f)
    if display:
        o3d.visualization.draw_geometries([chair],
                                          **_visualizer_params)
    
    return chair, data

def compute_distances(pcd, cropped_pcd, display = True):
    dists = np.asarray(pcd.compute_point_cloud_distance(cropped_pcd))
    index = np.where(dists > 0.01)[0]
    pcd_without_cropped = pcd.select_by_index(index)
    if display:
        o3d.visualization.draw_geometries([pcd_without_cropped],
                                          **_visualizer_params)
    
    return pcd_without_cropped

def dbscan_clustering(pcd, visualise = True):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.asarray(pcd.cluster_dbscan(eps = 0.02, min_points = 10, print_progress = True))

    max_labels = labels.max()
    print(f"Point clouds have {max_labels} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_labels if max_labels > 0 else 1))
    colors[colors < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if visualise:
        o3d.visualization.draw_geometries([pcd],
                                          **_visualizer_params)





if __name__ == "__main__":
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    
    downpcd = voxelize_and_display(pcd, 0.05, True)
    # estimate_normals_and_display(downpcd)

    # cropped_pcd, _ = crop_point_cloud(pcd, False)
    
    # compute_distances(pcd, cropped_pcd)

    dbscan_clustering(pcd)