import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler


def load_ply(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)
    points = np.asarray(point_cloud.points)
    return points


def downsample_point_cloud(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # voxel dwnsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)

    return downsampled_points


def compute_density(points, radius):
    # KDTree for efficient neighbor search
    kdtree = KDTree(points)

    # querying the KDTree to find the number of neighbors within the given radius for each point
    densities = kdtree.query_radius(points, r=radius, count_only=True)

    return densities


def color_points_by_density(points, densities, max_density):
    # normalize densities to the range [0, 1] so they can be used as colors
    # normalized_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities))
    normalized_densities = densities / max_density

    # colormap
    colors = plt.cm.plasma(normalized_densities)

    return colors


def cluster_pointsKMEANS(points, num_clusters):
    # K-Means clustering to group points into num clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(points)
    return labels


def cluster_pointsDBSCAN(points, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(
        StandardScaler().fit_transform(points)) + 1  # needs to be shifted to avoid negative label -1 for noise cluster
    return labels


def filter_biggest_cluster(points, labels):
    # find the label corresponding to the biggest cluster (based on number of points)
    biggest_cluster_label = np.argmax(np.bincount(labels))

    # keep only points belonging to the biggest cluster
    filtered_points = points[labels == biggest_cluster_label]

    return filtered_points


# TODO: remove, replaced by process_object
def process_placed_object(file_path, voxel_size=0.01, eps=0.1, min_samples=100, distance_threshold=0.3):
    points = load_ply(file_path)
    downsampled_points = downsample_point_cloud(points, voxel_size)

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)

    # perform clustering on the downsampled point cloud
    labels = cluster_pointsDBSCAN(downsampled_points, eps, min_samples)

    # filter out points that do not belong to the biggest cluster
    filtered_points = filter_biggest_cluster(downsampled_points, labels)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # removing the floor, fit plane using RANSAC
    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model

    # compute distance of each point from the plane
    distances = (a * filtered_points[:, 0] + b * filtered_points[:, 1] + c * filtered_points[:, 2] + d) / np.sqrt(
        a ** 2 + b ** 2 + c ** 2)

    below_indices = np.where(distances < -distance_threshold)[0]
    above_indices = np.where(distances > distance_threshold)[0]

    # determine which halfspace has more points
    if len(below_indices) > len(above_indices):
        visualization_indices = below_indices
    else:
        visualization_indices = above_indices

    visualization_points = filtered_points[visualization_indices]

    # color points based on density
    density_radius = 0.1
    densities = compute_density(visualization_points, density_radius)
    max_density = np.max(densities)
    colored_points = color_points_by_density(visualization_points, densities, max_density)

    visualization_pcd = o3d.geometry.PointCloud()
    visualization_pcd.points = o3d.utility.Vector3dVector(visualization_points)
    visualization_pcd.colors = o3d.utility.Vector3dVector(colored_points[:, :3])
    o3d.visualization.draw_geometries([visualization_pcd])

    return visualization_pcd


def process_object(file_path, voxel_size=0.01, eps=0.1, min_samples=100, distance_threshold=0.3, display=False):
    points = load_ply(file_path)

    downsampled_points = downsample_point_cloud(points, voxel_size)
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)

    labels = cluster_pointsDBSCAN(downsampled_points, eps, min_samples)
    filtered_points = filter_biggest_cluster(downsampled_points, labels)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model

    distances = (a * filtered_points[:, 0] + b * filtered_points[:, 1] + c * filtered_points[:, 2] + d) / np.sqrt(
        a ** 2 + b ** 2 + c ** 2)

    below_indices = np.where(distances < -distance_threshold)[0]
    above_indices = np.where(distances > distance_threshold)[0]

    if len(below_indices) > len(above_indices):
        visualization_indices = below_indices
    else:
        visualization_indices = above_indices

    visualization_points = filtered_points[visualization_indices]
    visualization_pcd = o3d.geometry.PointCloud()
    visualization_pcd.points = o3d.utility.Vector3dVector(visualization_points)
    # visualization_pcd.colors = o3d.utility.Vector3dVector(colored_points[:, :3])

    if display:
        # Visualize the result
        o3d.visualization.draw_geometries([visualization_pcd])

    return visualization_pcd


if __name__ == "__main__":
    pass
