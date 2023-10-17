import numpy as np
import open3d as o3d

if __name__ == '__main__':
    print("Testing IO for meshes ...")
    
    
    
    
    mesh = o3d.io.read_triangle_mesh("/Users/mario/Desktop/Project/reconstruction-deep-network/Desktop/Project/v1/scans/17DRP5sb8fy/new sub dir/house_segmentations/17DRP5sb8fy.ply")
    
    point_cloud = mesh.sample_points_uniformly(number_of_points=1000000)
    
    o3d.visualization.draw_geometries([point_cloud])
    
    # o3d.visualization.draw_geometries([mesh])
    
    
    
    







# Create a point cloud by sampling points from the mesh


# Save the point cloud to a .pcd file
# o3d.io.write_point_cloud("/path/to/your/point_cloud.pcd", point_cloud)






# Create a visualization window
