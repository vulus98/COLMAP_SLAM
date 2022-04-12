import os
from pathlib import Path
import numpy as np
import pycolmap
from src import features as feature_detector
from src import map_initialization, enums
from hloc.utils import viz_3d

# images = Path('data/frames/test1/')
images = Path('data/rgbd_dataset_freiburg2_xyz/rgb/')
outputs = Path('out/test1/')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

# FLANN is a nearest neighbour matching. Fast and less accurate.
# HAMMING returns the best match, accurate but slow.

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
    # apparently the first image you give for the matcher is the query and the second one the train
    return [match for match in matches if euc_dist_check(kp1[match.queryIdx].pt, kp2[match.trainIdx].pt)]
    # return [match for match in matches if euc_dist_check(kp1[match.trainIdx].pt, kp2[match.queryIdx].pt)]


def img_to_name(name, img):
    return name[:-4] + " " + str(img.tvec[0]) + " " + str(img.tvec[1]) + " " + str(img.tvec[2]) + " " + str(
        img.qvec[0]) + " " + str(img.qvec[1]) + " " + str(img.qvec[2]) + " " + str(img.qvec[3]) + "\n"


if __name__ == '__main__':
    currFrameIdx = 0
    keypointIdx = 0
    frameNames = os.listdir(images)

    # Assuming the frames are indexed
    frameNames.sort()

    frameNames = frameNames[:200]

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

    triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction)

    used_matcher = enums.Matchers.OrbHamming

    # TODO fix initialization
    #map_initialization.initialize_map(images, frameNames, reconstruction, graph, triangulator, options, camera, used_matcher)

    f = open(str(outputs / "estimation.txt"), "w")
    # f.write(img_to_name(currentKeyframe["name"], old_im))

    #currFrameIdx += 1
    #keypointIdx += 1
    while currFrameIdx < len(frameNames):
        if currFrameIdx % 15 == 0:
            kp, des = feature_detector.orb_detector(images / frameNames[currFrameIdx])
            detector = {
                "name": frameNames[currFrameIdx],
                "kp": kp,
                "des": des
            }

            if keypointIdx == 0:
                im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                                        tvec=[0, 0, 0])
                im.registered = True
                points2D = [keypoint.pt for keypoint in kp]
                im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
                reconstruction.add_image(im)
                reconstruction.add_point3D([0, 0, 0], pycolmap.Track(), np.zeros(3))
                graph.add_image(im.image_id, len(im.points2D))
            else:
                # Extracts all matches
                if used_matcher == enums.Matchers.OrbFlann:
                    matches, matchesMask = feature_detector.orb_matcher_FLANN(currentKeyframe, detector)
                    # good_matches = [matches[i][0] for i in range(len(matches)) if matchesMask[i][0] == 1]
                    # draw_matches_knn(currentKeyframe, detector, matches, matchesMask, indx=currentKeyframe["name"])
                    # matches = good_matches
                if used_matcher == enums.Matchers.OrbHamming:
                    matches = feature_detector.orb_matcher(currentKeyframe, detector)

                # Estimate Relative pose between the two images
                answer = pycolmap.two_view_geometry_estimation(
                    [currentKeyframe["kp"][match.queryIdx].pt for match in matches],
                    [detector["kp"][match.trainIdx].pt for match in matches],
                    camera,
                    camera
                )

                im = pycolmap.Image(id=keypointIdx, name=str(keypointIdx), camera_id=camera.camera_id,
                                    tvec=(old_im.tvec + answer["tvec"]), qvec=(old_im.qvec + answer["qvec"]))
                f.write(img_to_name(frameNames[currFrameIdx], im))
                points2D = [keypoint.pt for keypoint in kp]
                im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
                im.registered = True
                reconstruction.add_image(im)

                matches = [(match.queryIdx, match.trainIdx) for match in matches]
                matches = np.array(matches, dtype=np.uint32)

                # add image and correspondence to graph
                graph.add_image(im.image_id, len(im.points2D))
                graph.add_correspondences(old_im.image_id, im.image_id, matches)

            triangulator.triangulate_image(options, keypointIdx)

            currentKeyframe = detector
            old_im = im
            keypointIdx += 1
        currFrameIdx += 1

    num_completed_obs = triangulator.complete_all_tracks(options)
    # triangulator.merge_all_tracks(options)
    # triangulator.retriangulate(options)
    num_merged_obs = triangulator.merge_all_tracks(options)
    print(num_completed_obs)
    print(num_merged_obs)

    f.close()

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, reconstruction, min_track_length=0, color='rgb(255,0,0)')
    fig.show()
