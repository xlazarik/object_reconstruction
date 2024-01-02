import copy

import open3d as o3d
import numpy as np
from pointcloud_post_process import process_object


def test_prepare_pointclouds(filepath1, filepath2):
    cloud1 = o3d.io.read_point_cloud(filepath1)
    cloud2 = o3d.io.read_point_cloud(filepath2)

    cloud1 = process_object(filepath1, distance_threshold=0.05)
    cloud2 = process_object(filepath2, distance_threshold=0.0)

    # display original point clouds together
    # o3d.visualization.draw_geometries([cloud1, cloud2], window_name="Original Point Clouds")

    return cloud1, cloud2


def perform_ICP_registration(pcloud1, pcloud2):
    cloud1 = copy.deepcopy(pcloud1)
    cloud2 = copy.deepcopy(pcloud2)

    points1 = np.asarray(cloud1.points)
    points2 = np.asarray(cloud2.points)

    trans_init = np.identity(4)  # init transformation matrix (identity)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2)),
        max_correspondence_distance=0.05,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    # apply the transformation to align the point clouds
    cloud1.transform(reg_p2p.transformation)

    # visualize aligned results
    o3d.visualization.draw_geometries([cloud1, cloud2], window_name="Aligned Point Clouds")

    return cloud1, cloud2


def perform_RANSAC_fpfh_registration(pcloud1, pcloud2):
    cloud1 = copy.deepcopy(pcloud1)
    cloud2 = copy.deepcopy(pcloud2)

    points1 = np.asarray(cloud1.points)
    points2 = np.asarray(cloud2.points)

    radius_normal = 0.03
    radius_feature = 0.05  # radius for FPFH feature estimation

    # estimate normals
    cloud1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    cloud2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # compute fpfh features
    fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
        cloud1, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
        cloud2, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    distance_threshold = 0.1

    reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        cloud1,
        cloud2,
        source_feature=fpfh1,
        target_feature=fpfh2,
        mutual_filter=False,
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source=cloud1, target=cloud2, source_feature=fpfh1, target_feature=fpfh2,
    #     max_correspondence_distance=distance_threshold,
    #     mutual_filter=False,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     ransac_n=4,  # max_iteration
    #     checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #               o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    #     criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    # )

    # apply the transform
    cloud1.transform(reg_p2p.transformation)

    o3d.visualization.draw_geometries([cloud1, cloud2], window_name="Aligned Point Clouds")

    return cloud1, cloud2


####################################################################
if __name__ == "__main__":
    filepath_front_right = "C:/Users/vladi/Desktop/COLMAP-3.8-windows-cuda/WORKSPACE_matchbox/RAW_FRONT_RIGHT_MANUAL/dense/0/fused.ply"
    filepath_bottom = "C:/Users/vladi/Desktop/COLMAP-3.8-windows-cuda/WORKSPACE_matchbox/RAW_BOTTOM_HELD/dense/0/fused.ply"

    c1, c2 = test_prepare_pointclouds(filepath_front_right, filepath_bottom)
    tc1, tc2 = perform_RANSAC_fpfh_registration(c1, c2)
    perform_ICP_registration(tc1, tc2)
