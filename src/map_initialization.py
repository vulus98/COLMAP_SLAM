import time
import cv2 as cv

import pycolmap
import numpy as np
import cv2
import pyceres
from pathlib import Path
from hloc.utils import viz_3d

import slam
from src import features as feature_detector, enums, optimization


def initialize_map(img_pth, frameNames, reconstruction, graph, triangulator, traingulator_options, camera, map_points,
                   used_matcher=enums.Matchers.OrbHamming):
    debug = True
    currFrameIdx = 0
    kp_1, des_1 = feature_detector.detector(img_pth / frameNames[currFrameIdx], used_matcher) #, save=debug, out_pth=slam.outputs, name=(str(currFrameIdx) +  '.jpg'))
    # det_list = []
    detector1 = {
        "name": frameNames[currFrameIdx],
        "kp": kp_1,
        "des": des_1
    }
    # Rotation and translation of the first image
    R = np.eye(3)
    qr = pycolmap.rotmat_to_qvec(R)
    tv = [0, 0, 0]
    # For TUM freiburg2_xyz evaluation use image 0 with :
    # timestamp tx ty tz qx qy qz qw
    # 1311867170.4622 0.1163 -1.1498 1.4015 -0.5721 0.6521 -0.3565 0.3469
    # tv = [0.1163, -1.1498, 1.4015]
    # qr = [0.3469, -0.5721, 0.6521, -0.3565]
    # R = pycolmap.qvec_to_rotmat(qr)
    old_im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx),
                            camera_id=camera.camera_id, tvec=tv,
                            qvec=qr)  # , tvec=[0, 0, 0], qvec=[1, 0, 0, 0])
    old_im.registered = True
    points2D_1 = [keypoint.pt for keypoint in kp_1]
    old_im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D_1])
    reconstruction.add_image(old_im)
    graph.add_image(old_im.image_id, len(old_im.points2D))

    img_list = []
    for i in range(3):
        currFrameIdx += 7
        kp_2, des_2 = feature_detector.orb_detector(img_pth / frameNames[currFrameIdx]) # , save=debug, out_pth=slam.outputs, name=(str(currFrameIdx) + '.jpg'))
        detector2 = {
            "name": frameNames[currFrameIdx],
            "kp": kp_2,
            "des": des_2
        }

        # matches mask is empty for used_matcher=slam.Matchers.Hamming
        matches, matchesMask = feature_detector.matcher(detector1, detector2, used_matcher)

        # Estimate Relative pose between the two images
        answer = pycolmap.two_view_geometry_estimation(
            [detector1["kp"][match.queryIdx].pt for match in matches],
            [detector2["kp"][match.trainIdx].pt for match in matches],
            camera,
            camera
        )
        if debug:
            print("Relative pose estimation", answer["success"], ":", answer["configuration_type"])
        # TODO check if not degenerate etc.
        ''''
        .value("UNDEFINED", TwoViewGeometry::UNDEFINED)
        .value("DEGENERATE", TwoViewGeometry::DEGENERATE)
        .value("CALIBRATED", TwoViewGeometry::CALIBRATED)
        .value("UNCALIBRATED", TwoViewGeometry::UNCALIBRATED)
        .value("PLANAR", TwoViewGeometry::PLANAR)
        .value("PANORAMIC", TwoViewGeometry::PANORAMIC)
        .value("PLANAR_OR_PANORAMIC", TwoViewGeometry::PLANAR_OR_PANORAMIC)
        .value("WATERMARK", TwoViewGeometry::WATERMARK)
        .value("MULTIPLE", TwoViewGeometry::MULTIPLE);'''
        b = answer["success"]
        if b:
            # Rotation of the other images relative to the first one
            # If our first image is not placed at the origin
            R2 = R @ pycolmap.qvec_to_rotmat(answer["qvec"])
            qr2 = pycolmap.rotmat_to_qvec(R2)
            tv2 = tv + answer["tvec"]
            im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                                tvec=tv2, qvec=qr2)
            im.registered = True
            points2D_2 = [keypoint.pt for keypoint in kp_2]

            matches = [(match.queryIdx, match.trainIdx) for match in matches]
            matches = np.array(matches, dtype=np.uint32)
            matches = matches[np.array(answer["inliers"], dtype=bool)]
            im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D_2])
            reconstruction.add_image(im)
            # add image and correspondence to graph
            graph.add_image(im.image_id, len(im.points2D))
            graph.add_correspondences(old_im.image_id, im.image_id, matches)
            img_list.append(im.image_id)
            # det_list.append(detector2)

    max_reproj_error = 7  # 7
    max_angle_error = 2.0  # 2
    min_tri_angle = 1.5  # 1.5

    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = max_angle_error
    options.continue_max_angle_error = max_angle_error
    options.merge_max_reproj_error = max_reproj_error
    options.complete_max_reproj_error = max_reproj_error

    ret_a = triangulator.triangulate_image(options, old_im.image_id)

    fig1 = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig1, reconstruction, min_track_length=0, color='rgb(255,0,0)')

    # for id in img_list:
    # for id2 in img_list[(img_list.index(id) + 1):]:
    # matches mask is empty for used_matcher=slam.Matchers.Hamming
    #    matches, matchesMask = feature_detector.matcher(det_list[img_list.index(id)], det_list[img_list.index(id2)], used_matcher)
    #    matches = [(match.queryIdx, match.trainIdx) for match in matches]
    #    matches = np.array(matches, dtype=np.uint32)
    #    graph.add_correspondences(id, id2, matches)
    # ret_b = triangulator.triangulate_image(options, id)

    num_completed_obs = triangulator.complete_all_tracks(options)
    num_merged_obs = triangulator.merge_all_tracks(options)
    if debug:
        print("num_completed_obs", num_completed_obs)
        print("num_merged_obs", num_merged_obs)

    ret_f = reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle)
    if debug:
        print("Filtered", ret_f, "3D points out")
    optimization.global_BA(reconstruction, skip_pose=[old_im.image_id])

    # triangulator.retriangulate(options)

    viz_3d.plot_reconstruction(fig1, reconstruction, min_track_length=0, color='rgb(0,255,0)')
    # fig1.show()

    # Fill the map_points
    old_im = reconstruction.find_image_with_name(str(old_im.image_id))


    ret_a = reconstruction.num_points3D()
    if debug:
        print("Initial traingulation yielded", ret_a, "3D points")
    return (ret_a > 100), currFrameIdx


# Not working, should show the difference between the keypoint and the reprojection back to the image of the estimated
# 3D point corresponding to the keypoint
def draw_reprojection(reconstruction, camera):
    # For debugging and visualization
    for img in reconstruction.images.values():
        pts2D = img.get_valid_points2D()[:10]
        kp = [pt.xy for pt in pts2D]
        pts3D = [reconstruction.points3D[pt.point3D_id].xyz for pt in pts2D]
        repr = img.project(pts3D)
        h_repr = [[r[0] * camera.width, r[1] * camera.height] for r in repr]
        cv_img = cv.imread(str(img_pth / frameNames[img.image_id]), 0)
        img2 = cv.drawKeypoints(cv_img, kp, None, color=(0, 255, 0), flags=0)
        img2 = cv.drawKeypoints(img2, h_repr, None, color=(0, 0, 255), flags=0)
        name = str(img.name) + ".jpg"
        cv.imwrite(str(slam.outputs / 'images/reprojection' / name), img2)