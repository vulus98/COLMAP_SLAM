import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2 as cv
import numpy as np

import pycolmap

import hloc
from hloc import reconstruction, extract_features, match_features
from hloc.utils import viz_3d

images = Path('data/frames/test1/')
outputs = Path('out/test1/')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'


def orb_detector(img_pth, save=False, name='orb_out.jpg'):
    img = cv.imread(str(img_pth), 0)
    # img = Image.open(img_pth)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
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


def draw_matches(current_keyframe, _detector, _matches, indx=0):
    img1 = cv.imread(str(images / current_keyframe["name"]), 0)
    img2 = cv.imread(str(images / _detector["name"]), 0)
    # Draw first n matches.
    img3 = cv.drawMatches(img1, current_keyframe["kp"], img2, _detector["kp"], _matches[:10], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    name = "usedMatch_" + str(indx) + ".jpg"
    result = cv.imwrite(str(outputs / 'images/matcher' / name), img3)


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

    keyframes = []

    kp, des = orb_detector(images / frameNames[currFrameIdx])
    currentKeyframe = {
        "name": frameNames[currFrameIdx],
        "kp": kp,
        "des": des
    }
    keyframes.append(currentKeyframe)

    # width = 100
    # height = 100
    # focal_length = 1
    # cx = 0
    # cy = 0
    # camera = pycolmap.Camera(
    # model='SIMPLE_PINHOLE',
    # width=width,
    # height=height,
    # params=[focal_length, cx, cy],
    # )

    camera = pycolmap.infer_camera_from_image(images / frameNames[currFrameIdx])
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(camera)

    # model = pycolmap.Reconstruction.main()
    # sfm_dir.mkdir(parents=True, exist_ok=True)
    # database = sfm_dir / 'database.db'
    # hloc.reconstruction.create_empty_db(database)
    # pycolmap.import_images(database, images, pycolmap.CameraMode.AUTO)

    old_im = pycolmap.Image(id=keypointIdx, name=str(keypointIdx), camera_id=camera.camera_id, tvec=[0, 0, 0])
    old_im.registered = True
    reconstruction.add_image(old_im)
    reconstruction.add_point3D([0, 0, 0], pycolmap.Track(), np.zeros(3))

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

    #hloc.extract_features.main(hloc.extract_features.confs['superpoint_aachen'], images,
    #                           image_list=frameNames[currFrameIdx], feature_path=features)
    currFrameIdx += 1
    keypointIdx += 1
    while currFrameIdx < len(frameNames):
        kp, des = orb_detector(images / frameNames[currFrameIdx])
        detector = {
            "name": frameNames[currFrameIdx],
            "kp": kp,
            "des": des
        }
        # Extracts all matches
        matches = orb_matcher(currentKeyframe, detector)

        # 8 Just a chosen constant (at least 4 are needed for Homography)
        # if len(usedMatches) > 4:
        if currFrameIdx % 5 == 0:
            # Estimate RElative pose between the two images
            # TODO check type as in: https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.h#L47-L67
            answer = pycolmap.two_view_geometry_estimation(
                [currentKeyframe["kp"][match.queryIdx].pt for match in matches],
                [detector["kp"][match.trainIdx].pt for match in matches],
                camera,
                camera
            )

            # TODO make 3D point reconstruction

            im = pycolmap.Image(id=keypointIdx, name=str(keypointIdx), camera_id=camera.camera_id, tvec=(old_im.tvec + answer["tvec"]), qvec=(old_im.qvec + answer["qvec"]))
            #im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)])
            im.registered = True
            reconstruction.add_image(im)
            # reconstruction.add_point3D(old_im.tvec + answer["tvec"], pycolmap.Track(), np.zeros(3))

            keyframes.append(detector)
            currentKeyframe = detector
            old_im = im
            keypointIdx += 1
            # print(currFrameIdx)
        currFrameIdx += 1

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, reconstruction, min_track_length=0, color='rgb(255,0,0)')
    fig.show()
    a = 0
