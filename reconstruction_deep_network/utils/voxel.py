import numpy as np
import open3d as o3d
from reconstruction_deep_network.utils.visualization import _visualizer_params

def voxelize_and_display(pcd, size, display=True):
    # downpcd = pcd.voxel_down_sample(voxel_size=size)
    # if display:
    #     o3d.visualization.draw_geometries([downpcd], **_visualizer_params)
    # return downpcd
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