import os
from pathlib import Path
import numpy as np
import pycolmap
from src import features as feature_detector
from src import map_initialization, enums, optimization
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

# Output corresponding to the evaluation of the TUM-RGBD dataset
def img_to_name(name, img):
    return name[:-4] + " " + str(img.tvec[0]) + " " + str(img.tvec[1]) + " " + str(img.tvec[2]) + " " + str(
        img.qvec[0]) + " " + str(img.qvec[1]) + " " + str(img.qvec[2]) + " " + str(img.qvec[3]) + "\n"


# Extracts from the matches of a query (as an detector dict) all the corresponding valid 3D points in the keyframe
def match_to_3D_correspondences(query_detector, keyframe_img, matches):
    query_2D = []
    keyframe_3D = []
    for match in matches:
        id_3D = keyframe_img.points2D[match.queryIdx].point3D_id
        if id_3D >= 0 and id_3D < 18446744073709551615:
            query_2D.append(query_detector["kp"][match.trainIdx].pt)
            a = reconstruction
            keyframe_3D.append(reconstruction.points3D[id_3D].xyz)
    return query_2D, keyframe_3D


if __name__ == '__main__':
    frameNames = os.listdir(images)

    # Assuming the frames are indexed
    frameNames.sort()

    frameNames = frameNames[:200]

    # camera = pycolmap.infer_camera_from_image(images / frameNames[currFrameIdx])
    camera = pycolmap.Camera(
        model='PINHOLE',
        width=640,
        height=480,
        params=[525, 525, 319.5, 239.5],
    )
    camera.camera_id = 0
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(camera)
    graph = pycolmap.CorrespondenceGraph()

    max_reproj_error = 7.0
    max_angle_error = 2.0
    min_tri_angle = 0.5

    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = max_angle_error
    options.continue_max_angle_error = max_angle_error
    options.merge_max_reproj_error = max_reproj_error
    options.complete_max_reproj_error = max_reproj_error

    triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction)

    used_matcher = enums.Matchers.OrbHamming

    sucess, currFrameIdx, currentKeyframe = map_initialization.initialize_map(images, frameNames, reconstruction, graph,
                                                                              triangulator, options, camera,
                                                                              used_matcher)
    keypointIdx = currFrameIdx
    currFrameIdx += 1

    old_im = reconstruction.find_image_with_name(str(keypointIdx))

    f = open(str(outputs / "estimation.txt"), "w")
    # f.write(img_to_name(currentKeyframe["name"], old_im))

    while currFrameIdx < len(frameNames):
        if currFrameIdx % 1 == 0:
            kp, des = feature_detector.orb_detector(images / frameNames[currFrameIdx])
            detector = {
                "name": frameNames[currFrameIdx],
                "kp": kp,
                "des": des
            }

            # Extracts all matches
            matches, matchesMask = feature_detector.matcher(currentKeyframe, detector, used_matcher)

            query_pts, keyframe_pts = match_to_3D_correspondences(detector, old_im, matches)
            # Estimate absolute pose of the query image
            answer = pycolmap.absolute_pose_estimation(query_pts,
                                                       keyframe_pts,
                                                       camera, max_error_px=12.0)

            if answer["success"]:
                print("Frame", currFrameIdx, "sucess")
                im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                                    tvec=(answer["tvec"]), qvec=(answer["qvec"]))
                f.write(img_to_name(frameNames[currFrameIdx], im))
                points2D = [keypoint.pt for keypoint in kp]
                im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
                im.registered = True
                reconstruction.add_image(im)

                matches = [(match.queryIdx, match.trainIdx) for match in matches]
                matches = np.array(matches, dtype=np.uint32)
                # matches = matches[np.array(answer["inliers"], dtype=bool)]

                # add image and correspondence to graph
                graph.add_image(im.image_id, len(im.points2D))
                graph.add_correspondences(old_im.image_id, im.image_id, matches)

                num_tri = triangulator.triangulate_image(options, im.image_id)
                print("triangulated", num_tri, " new 3D points")
                num_completed_obs = triangulator.complete_all_tracks(options)
                num_merged_obs = triangulator.merge_all_tracks(options)

                # optimization.motion_only_BA(reconstruction, [old_im.image_id, im.image_id])
                # optimization.local_BA(reconstruction, [old_im.image_id, im.image_id])

                #print(num_completed_obs)
                #print(num_merged_obs)

                currentKeyframe = detector
                #old_im = im
                # for some reason we have to reload the iamge s.t. ther points are traingualted?
                old_im = reconstruction.find_image_with_name(str(im.name))
                keypointIdx += 1
            else:
                print("Frame ", currFrameIdx, "failure: not able to estimate absolute pose")
        currFrameIdx += 1

    # num_completed_obs = triangulator.complete_all_tracks(options)
    # triangulator.merge_all_tracks(options)
    # triangulator.retriangulate(options)
    # num_merged_obs = triangulator.merge_all_tracks(options)
    # print(num_completed_obs)
    # print(num_merged_obs)

    f.close()

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, reconstruction, min_track_length=0, color='rgb(255,0,0)')
    fig.show()
