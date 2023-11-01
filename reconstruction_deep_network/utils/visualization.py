"""
Visualization utils
"""

import open3d as o3d

_visualizer_params = {
    "zoom": 0.3412,
    "front": [0.4257, -0.2125, -0.8795],
    "lookat": [2.6172, 2.0475, 1.532],
    "up": [-0.0694, -0.9768, 0.2024]
}

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

