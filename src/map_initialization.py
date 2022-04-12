import pycolmap
import numpy as np
from pathlib import Path

from src import features as feature_detector, enums

def initialize_map(img_pth, frameNames, reconstruction, graph, triangulator, traingulator_options, camera, used_matcher=enums.Matchers.OrbHamming):
    currFrameIdx = 0
    kp, des = feature_detector.detector(img_pth / frameNames[currFrameIdx], used_matcher)
    detector1 = {
        "name": frameNames[currFrameIdx],
        "kp": kp,
        "des": des
    }

    old_im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id, tvec=[0, 0, 0])
    old_im.registered = True
    points2D = [keypoint.pt for keypoint in kp]
    old_im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
    reconstruction.add_image(old_im)
    reconstruction.add_point3D([0, 0, 0], pycolmap.Track(), np.zeros(3))

    graph.add_image(old_im.image_id, len(old_im.points2D))

    b = False
    while not b:
        currFrameIdx += 15
        kp, des = feature_detector.orb_detector(img_pth / frameNames[currFrameIdx])
        detector2 = {
            "name": frameNames[currFrameIdx],
            "kp": kp,
            "des": des
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
        print(answer["success"])
        b = answer["success"]
    im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                        tvec=answer["tvec"], qvec=answer["qvec"])
    points2D = [keypoint.pt for keypoint in kp]
    im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
    im.registered = True
    reconstruction.add_image(im)

    matches = [(match.queryIdx, match.trainIdx) for match in matches]
    matches = np.array(matches, dtype=np.uint32)

    # add image and correspondence to graph
    graph.add_image(im.image_id, len(im.points2D))
    graph.add_correspondences(old_im.image_id, im.image_id, matches)

    triangulator.triangulate_image(traingulator_options, 0)
    triangulator.triangulate_image(traingulator_options, currFrameIdx)
