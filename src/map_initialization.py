import pycolmap
import numpy as np
import cv2
import pyceres
from pathlib import Path

from src import features as feature_detector, enums, optimization


def initialize_map(img_pth, frameNames, reconstruction, graph, triangulator, traingulator_options, camera,
                   used_matcher=enums.Matchers.OrbHamming):
    currFrameIdx = 0
    kp_1, des_1 = feature_detector.detector(img_pth / frameNames[currFrameIdx], used_matcher)
    # det_list = []
    detector1 = {
        "name": frameNames[currFrameIdx],
        "kp": kp_1,
        "des": des_1
    }
    # det_list.append(detector1)
    old_im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx),
                            camera_id=camera.camera_id, tvec=[0, 0, 0], qvec=[1, 0, 0, 0])  # , tvec=[0, 0, 0], qvec=[1, 0, 0, 0])
    old_im.registered = True
    points2D_1 = [keypoint.pt for keypoint in kp_1]
    old_im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D_1])
    reconstruction.add_image(old_im)
    graph.add_image(old_im.image_id, len(old_im.points2D))

    img_list = []
    for i in range(3):
        currFrameIdx += 10
        kp_2, des_2 = feature_detector.orb_detector(img_pth / frameNames[currFrameIdx])
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
            im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                                tvec=answer["tvec"], qvec=answer["qvec"])
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

    max_reproj_error = 7
    max_angle_error = 2.0
    min_tri_angle = 1.5

    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = max_angle_error
    options.continue_max_angle_error = max_angle_error
    options.merge_max_reproj_error = max_reproj_error
    options.complete_max_reproj_error = max_reproj_error

    ret_a = triangulator.triangulate_image(options, old_im.image_id)
    # for id in img_list:
        #for id2 in img_list[(img_list.index(id) + 1):]:
            # matches mask is empty for used_matcher=slam.Matchers.Hamming
        #    matches, matchesMask = feature_detector.matcher(det_list[img_list.index(id)], det_list[img_list.index(id2)], used_matcher)
        #    matches = [(match.queryIdx, match.trainIdx) for match in matches]
        #    matches = np.array(matches, dtype=np.uint32)
        #    graph.add_correspondences(id, id2, matches)
        #ret_b = triangulator.triangulate_image(options, id)

    num_completed_obs = triangulator.complete_all_tracks(options)
    num_merged_obs = triangulator.merge_all_tracks(options)

    print("num_completed_obs", num_completed_obs)
    print("num_merged_obs", num_merged_obs)

    ret_f = reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle)
    optimization.global_BA(reconstruction, skip_pose=[old_im.image_id])

    return (ret_a > 100), currFrameIdx, detector2
