import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2 as cv
import numpy as np

import pycolmap
from matplotlib import pyplot as plt
import hloc
from hloc import reconstruction, extract_features, match_features
from hloc.utils import viz_3d

images = Path('data/test1/images/')
outputs = Path('output/test1/')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

# FLANN is a nearest neighbour matching. Fast and less accurate.
# HAMMING returns the best match, accurate but slow.

MATCHER = "FLANN" # or HAMMING



def orb_detector(img_pth, save=False, name='orb_out.jpg'):
    img = cv.imread(str(img_pth), 0)
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=2000)
    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(img, None)

    if save:
        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        result = cv.imwrite(str(outputs / 'images/detector' / name), img2)

    return kp, des


def orb_matcher(keypoint, query, save=False, name='orb_out.jpg'):
    kp1 = keypoint["kp"]
    des1 = keypoint["des"]
    kp2 = query["kp"]
    des2 = query["des"]

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    if save:
        draw_matches(keypoint, query, matches)
    return matches


def orb_matcher_FLANN(keypoint, query):
    des1 = keypoint["des"]
    des2 = query["des"]
    # Nearest neighbour matching
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # If they are both equidistance, the ratio will be 1. Ambiguous, so discard.
            matchesMask[i] = [1, 0]



    return matches, matchesMask


def draw_matches(current_keyframe, _detector, _matches, indx=0):
    img1 = cv.imread(str(images / current_keyframe["name"]), 0)
    img2 = cv.imread(str(images / _detector["name"]), 0)
    # Draw first n matches.
    img3 = cv.drawMatches(img1, current_keyframe["kp"], img2, _detector["kp"], _matches[:10], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    name = "usedMatch_" + str(indx) + ".jpg"
    result = cv.imwrite(str(outputs / 'images/matcher' / name), img3)


def draw_matches_knn(current_keyframe, _detector, _matches, indx=0):
    img1 = cv.imread(str(images / current_keyframe["name"]), 0)
    img2 = cv.imread(str(images / _detector["name"]), 0)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, current_keyframe["kp"], img2, _detector["kp"], matches, None, **draw_params)
    name = "usedMatch_" + str(indx) + ".jpg"
    cv.imwrite(str(outputs / 'images/matcher' / name), img3)


def euc_dist_check(pt1, pt2):
    dist = (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2
    if (2 ** 2) < dist < (4 ** 2):
        a = 0
    return (0) < dist < (2 ** 2)


def good_matches(keypoint, query, matches):
    kp1 = keypoint["kp"]
    # des1 = keypoint["des"]
    kp2 = query["kp"]
    # des2 = query["des"]
    # appearently the firsst image you give for the matcher is the query and the secod one the train
    return [match for match in matches if euc_dist_check(kp1[match.queryIdx].pt, kp2[match.trainIdx].pt)]
    # return [match for match in matches if euc_dist_check(kp1[match.trainIdx].pt, kp2[match.queryIdx].pt)]


def orb_kp_to_keypints(kps):
    a = [np.ndarray[np.float64[2, 1]]]
    # a.extend([keypoint.pt for keypoint in kps])
    return np.ndarray([keypoint.pt for keypoint in kps])


if __name__ == '__main__':
    currFrameIdx = 0
    keypointIdx = 0
    frameNames = os.listdir(images)

    # Assuming the frames are indexed
    frameNames.sort()

    camera = pycolmap.infer_camera_from_image(images / frameNames[currFrameIdx])
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(camera)
    graph = pycolmap.CorrespondenceGraph()

    max_reproj_error = 10.0
    max_angle_error = 5.0
    min_tri_angle = 5

    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = max_angle_error
    options.continue_max_angle_error = max_angle_error
    options.merge_max_reproj_error = max_reproj_error
    options.complete_max_reproj_error = max_reproj_error

    # model = pycolmap.Reconstruction.main()
    # sfm_dir.mkdir(parents=True, exist_ok=True)
    # database = sfm_dir / 'database.db'
    # hloc.reconstruction.create_empty_db(database)
    # pycolmap.import_images(database, images, pycolmap.CameraMode.AUTO)

    # keyframes = []

    kp, des = orb_detector(images / frameNames[currFrameIdx])
    currentKeyframe = {
        "name": frameNames[currFrameIdx],
        "kp": kp,
        "des": des
    }
    # keyframes.append(currentKeyframe)

    old_im = pycolmap.Image(id=keypointIdx, name=str(keypointIdx), camera_id=camera.camera_id, tvec=[0, 0, 0])
    old_im.registered = True
    points2D = [keypoint.pt for keypoint in kp]
    old_im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
    reconstruction.add_image(old_im)
    reconstruction.add_point3D([0, 0, 0], pycolmap.Track(), np.zeros(3))

    graph.add_image(old_im.image_id, len(old_im.points2D))

    # pycolmap.pose
    # fig = viz_3d.init_figure()
    # viz_3d.plot_reconstruction(fig, reconstruction, min_track_length=0, color='rgb(255,0,0)')
    # fig.show()

    # img = pycolmap.Image(name=frameNames[currFrameIdx],
    #                     keypoints=orb_kp_to_keypints(kp),
    #                     cameraid=0)
    # img.SetName(frameNames[currFrameIdx])
    # img.SetPoints2D(kp)
    # img.SetImageId(0)
    # reconstruction.add_image(img)

    # hloc.extract_features.main(hloc.extract_features.confs['superpoint_aachen'], images,
    #                           image_list=frameNames[currFrameIdx], feature_path=features)

    triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction)
    # triangulator.triangulate_image(options, keypointIdx)

    currFrameIdx += 1
    keypointIdx += 1
    while currFrameIdx < len(frameNames):
        # 8 Just a chosen constant (at least 4 are needed for Homography)
        # if len(usedMatches) > 4:
        kp, des = orb_detector(images / frameNames[currFrameIdx])
        detector = {
            "name": frameNames[currFrameIdx],
            "kp": kp,
            "des": des
        }

        # Extracts all matches

        matches, matchesMask = orb_matcher_FLANN(currentKeyframe, detector)
        good_matches = [matches[i][0] for i in range(len(matches)) if matchesMask[i][0] == 1]
        draw_matches_knn(currentKeyframe, detector, matches, indx=currentKeyframe["name"])
        matches = good_matches

        # Estimate Relative pose between the two images
        answer = pycolmap.two_view_geometry_estimation(
            [currentKeyframe["kp"][match.queryIdx].pt for match in matches],
            [detector["kp"][match.trainIdx].pt for match in matches],
            camera,
            camera
        )

        im = pycolmap.Image(id=keypointIdx, name=str(keypointIdx), camera_id=camera.camera_id,
                            tvec=(old_im.tvec + answer["tvec"]), qvec=(old_im.qvec + answer["qvec"]))
        # im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)])
        points2D = [keypoint.pt for keypoint in kp]
        im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
        im.registered = True
        reconstruction.add_image(im)
        # reconstruction.add_point3D(old_im.tvec + answer["tvec"], pycolmap.Track(), np.zeros(3))

        matches = [(match.trainIdx, match.queryIdx) for match in matches]
        matches = np.array(matches, dtype=np.uint32)
        # add image and correspondence to graph
        graph.add_image(im.image_id, len(im.points2D))
        graph.add_correspondences(old_im.image_id, im.image_id, matches)

        triangulator.triangulate_image(options, keypointIdx)

        # keyframes.append(detector)
        currentKeyframe = detector
        old_im = im
        keypointIdx += 1
        # print(currFrameIdx)
        currFrameIdx += 1

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, reconstruction, min_track_length=0, color='rgb(255,0,0)')
    fig.show()
    a = 0
