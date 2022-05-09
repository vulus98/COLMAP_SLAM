import time
import cv2 as cv

import pycolmap
import numpy as np
import cv2
import pyceres
from pathlib import Path
from hloc.utils import viz_3d

from src import features, enums, optimization

cv.namedWindow('image', cv.WINDOW_NORMAL)

def initialize_map(img_pth, frameNames, reconstruction, graph, triangulator, triangulator_options, camera, map_points,
                   extractor, matcher,
                   used_extractor=enums.Extractors.ORB,
                   used_matcher=enums.Matchers.OrbHamming):
    debug = True
    currFrameIdx = 0
    kp1, detector1 = features.detector(img_pth, frameNames[currFrameIdx], extractor, used_extractor) # , save=debug, out_pth=slam.outputs, name=(str(currFrameIdx) +  '.jpg'))
    # det_list = []
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
    first_image = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx),
                            camera_id=camera.camera_id, tvec=tv,
                            qvec=qr)
    first_image.registered = True
    points2D_1 = [pycolmap.Point2D(keypoint.pt) for keypoint in kp1]
    first_image.points2D = pycolmap.ListPoint2D(points2D_1)
    reconstruction.add_image(first_image)
    graph.add_image(first_image.image_id, len(first_image.points2D))

    matches_for_visualization = []
    img_list = []

    first_frame_read_image = cv.imread(str(img_pth / frameNames[0]))
    for kp in kp1:
        cv.circle(first_frame_read_image, (int(round(kp.pt[0])), int(round(kp.pt[1]))), color=(0, 255, 0), radius=3)
    cv.imshow('image', first_frame_read_image)
    cv.waitKey()

    last_kp = kp1
    last_detector = detector1
    last_image = first_image

    i=0
    while i<4:
        currFrameIdx += 2
        kp2, detector2 = features.detector(img_pth, frameNames[currFrameIdx], extractor, used_extractor) #, save=debug, out_pth=slam.outputs, name=(str(currFrameIdx) + '.jpg'))

        constant_tvec = []
        # matches mask is empty for used_matcher=slam.Matchers.Hamming
        matches = features.matcher(last_detector, detector2, matcher, used_matcher)

        two_view_geometry_options = pycolmap.TwoViewGeometryOptions()
        two_view_geometry_options.ransac.min_num_trials = 100

        # Estimate Relative pose between the two images
        answer = pycolmap.two_view_geometry_estimation(
            [last_kp[match.queryIdx].pt for match in matches],
            [kp2[match.trainIdx].pt for match in matches],
            camera,
            camera,
        )

        # TODO: The return dictionary contains the relative pose, inlier mask, as well as the type of camera configuration, such as degenerate or planar.
        # The answer variable above can also be fed some options, but here there are no such options provided.
        # Examples of options are:
        # min_num_inlier, ransac options, etc.

        # TODO: We should also check the translation vector/ optical flow etc. Preferrably the optical flow. For now,
        # try to visualize the images and the number of matches, number of keypoints selected.

        if debug:
            print("Relative pose estimation", answer["success"], ":", answer["configuration_type"])
        if answer['configuration_type'] != 2:
            print(f"Skipping index {currFrameIdx} because the config type is not 2. It is {answer['configuration_type']}")
            continue
        # TODO check if not degenerate etc.
        ''''
        .value("UNDEFINED", TwoViewGeometry::UNDEFINED) 0
        .value("DEGENERATE", TwoViewGeometry::DEGENERATE) 1 Degenerate configuration, eg. no overlap or not enough inliers
        .value("CALIBRATED", TwoViewGeometry::CALIBRATED) 2 essential matrix
        .value("UNCALIBRATED", TwoViewGeometry::UNCALIBRATED) 3 fundamental matrix
        .value("PLANAR", TwoViewGeometry::PLANAR) 4 homography, planar scene with baseline
        .value("PANORAMIC", TwoViewGeometry::PANORAMIC) 5 homography, pure rotation without baseline
        .value("PLANAR_OR_PANORAMIC", TwoViewGeometry::PLANAR_OR_PANORAMIC) 6 homography, planar or panoramic
        .value("WATERMARK", TwoViewGeometry::WATERMARK) 7 watermark, pure 2d translation in image borders.
        .value("MULTIPLE", TwoViewGeometry::MULTIPLE);8 multi-model configuration, i.e. the inlier matches result from multiple individual, non-degenerate configurations.
        '''
        # Baseline b > ratio * z (depth)

        # Assuming no rotation:
        # Median over the image points
        # (u - u´) / f
        # At least a 100 points are close, a lot of flow
        # Thresholdig the flow: (f * t) / z
        #                          focal length, tvec, z depth
        # This gives tvec / z = delta_u (2D point in the images) / f < 10%
        b = answer["success"] #TODO: being a success is not enough, remember to input good options to the two view estimation.
        if b:
            # path = img_pth/frameNames[0]
            # R^t
            # -t
            # Rotation of the other images relative to the first one
            # If our first image is not placed at the origin
            # https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes
            # R_answ = R_2^T @ R_1  ==>  R_2 @ R_answ = R_1  ==> R_2 = R_answ = R_1 @ R_answ^T
            # t_answ = R^T @ (tv - tv2) ==> R @ t_answ = tv - tv2 ==> tv2 = tv - R @ t_answ
            # R2 = R @ np.transpose(pycolmap.qvec_to_rotmat(answer["qvec"]))
            # qr2 = pycolmap.rotmat_to_qvec(R2)
            # tv2 = tv - R @ answer["tvec"]

            tv2 = answer["tvec"]
            qr2 = answer["qvec"]
            im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                                tvec=tv2, qvec=qr2)
            im.registered = True
            points2D_2 = [pycolmap.Point2D(keypoint.pt) for keypoint in kp2]

            matches = [(match.queryIdx, match.trainIdx) for match in matches]
            matches = np.array(matches, dtype=np.uint32)
            matches = matches[np.array(answer["inliers"], dtype=bool)]

            matches_kp_for_viz = [(last_kp[curr].pt, kp2[first].pt) for curr, first in matches]

            img = cv.imread(str(img_pth / frameNames[currFrameIdx]))
            for pt1, pt2 in matches_kp_for_viz:
                u1, v1 = map(lambda x: int(round(x)), pt1)
                u2, v2 = map(lambda x: int(round(x)), pt2)

                cv.circle(img, (u1, v1), color=(0,255,0), radius=3)
                cv.line(img, (u1,v1), (u2,v2), color=(255,0,0))

            cv.imshow('image', img)
            cv.waitKey()
            im.points2D = pycolmap.ListPoint2D(points2D_2)
            reconstruction.add_image(im)
            # add image and correspondence to graph
            graph.add_image(im.image_id, len(im.points2D))
            graph.add_correspondences(last_image.image_id, im.image_id, matches)
            img_list.append(im.image_id)
            constant_tvec.append(im.image_id)
            # det_list.append(detector2)

            options = pycolmap.IncrementalTriangulatorOptions()
            print(f"Triangulated: {triangulator.triangulate_image(options, im.image_id)}")
            print(f"Number of completed observations: {triangulator.complete_image(options, last_image.image_id)}")
            print(f"Number of completed tracks: {triangulator.complete_all_tracks(options)}")
            print(f"Number of merged tracks: {triangulator.merge_all_tracks(options)}")
            print(f"Number of 3D points filtered out: {reconstruction.filter_all_points3D(4, 1.5)}")
            print(f"Reconstruction Summary: {reconstruction.summary()}")


            if i>=10:
                # Bundle Adjustment
                ba = optimization.BundleAdjuster(reconstruction)
                ba.global_BA()


            last_kp = kp2
            last_detector = detector2
            last_image = im

            i+=1

    max_reproj_error = 4  # 7
    max_angle_error = 2.0  # 2
    min_tri_angle = 1.5  # 1.5

    # options = pycolmap.IncrementalTriangulatorOptions()
    # options.create_max_angle_error = max_angle_error
    # options.continue_max_angle_error = max_angle_error
    # options.merge_max_reproj_error = max_reproj_error
    # options.complete_max_reproj_error = max_reproj_error
    # options.ignore_two_view_track = False

    # print(triangulator.triangulate_image(options, first_image.image_id))
    # print(triangulator.complete_image(options, first_image.image_id))
    fig1 = viz_3d.init_figure()
    # viz_3d.plot_reconstruction(fig1, reconstruction, min_track_length=0, color='rgb(255,0,0)', name='no optimization')

    # for id in img_list:
    # for id2 in img_list[(img_list.index(id) + 1):]:
    # matches mask is empty for used_matcher=slam.Matchers.Hamming
    #    matches, matchesMask = feature_detector.matcher(det_list[img_list.index(id)], det_list[img_list.index(id2)], used_matcher)
    #    matches = [(match.queryIdx, match.trainIdx) for match in matches]
    #    matches = np.array(matches, dtype=np.uint32)
    #    graph.add_correspondences(id, id2, matches)
    # ret_b = triangulator.triangulate_image(options, id)

    # num_completed_obs = triangulator.complete_all_tracks(options)
    # num_merged_obs = triangulator.merge_all_tracks(options)
    # if debug:
    #    print("num_completed_obs", num_completed_obs)
    #    print("num_merged_obs", num_merged_obs)

    # ret_f = reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle)
    # if debug:
    #     print("Filtered", ret_f, "3D points out")

    # Bundle Adjustment
    # ba = optimization.BundleAdjuster(reconstruction)
    # ba.global_BA()

    # triangulator.retriangulate(options)

    # ret_f = reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle)
    # if debug:
    #     print("Filtered", ret_f, "3D points out")

    # fig1.show()

    # Fill the map_points
    # old_im = reconstruction.find_image_with_name(str(old_im.image_id))

    # print(f"Retriangulation: {triangulator.retriangulate(options)}")
    ret_a = reconstruction.num_points3D()
    if debug:
        print("Initial triangulation yielded", ret_a, "3D points")
    viz_3d.plot_reconstruction(fig1, reconstruction, min_track_length=0, color='rgb(0,255,0)', name='global BA')

    return (ret_a > 50), currFrameIdx


# Not working, should show the difference between the keypoint and the reprojection back to the image of the estimated
# 3D point corresponding to the keypoint
# def draw_reprojection(reconstruction, camera):
#     # For debugging and visualization
#     for img in reconstruction.images.values():
#         pts2D = img.get_valid_points2D()[:10]
#         kp = [pt.xy for pt in pts2D]
#         pts3D = [reconstruction.points3D[pt.point3D_id].xyz for pt in pts2D]
#         repr = img.project(pts3D)
#         h_repr = [[r[0] * camera.width, r[1] * camera.height] for r in repr]
#         cv_img = cv.imread(str(img_pth / frameNames[img.image_id]), 0)
#         img2 = cv.drawKeypoints(cv_img, kp, None, color=(0, 255, 0), flags=0)
#         img2 = cv.drawKeypoints(img2, h_repr, None, color=(0, 0, 255), flags=0)
#         name = str(img.name) + ".jpg"
#         cv.imwrite(str(slam.outputs / 'images/reprojection' / name), img2)