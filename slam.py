import os
from pathlib import Path
import numpy as np
import pycolmap
from src import features
from src import map_initialization, enums, optimization
from hloc.utils import viz_3d
import open3d as o3d



# images = Path('data/frames/test1/')
images = Path('data/rgbd_dataset_freiburg2_xyz/rgb/')
outputs = Path('out/test1/')
#sfm_pairs = outputs / 'pairs-sfm.txt'
#loc_pairs = outputs / 'pairs-loc.txt'
#sfm_dir = outputs / 'sfm'
#features = outputs / 'features.h5'
#matches = outputs / 'matches.h5'
exports = outputs / 'reconstruction.ply'
#points_exports = outputs / 'reconstruction_points.ply'

# if false, the hloc viz will be used
USE_OPEN_3D = True


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
    return name[:-4] + " " + str(img.tvec[0]) + " " + str(img.tvec[1]) + " " + str(img.tvec[2]) + " " \
           + str(img.qvec[1]) + " " + str(img.qvec[2]) + " " + str(img.qvec[3]) + " " + str(img.qvec[0]) + "\n"


# Extracts from the matches of a query (as an detector dict) all the corresponding valid 3D points in the keyframe
def match_to_3D_correspondences(query_detector_kps, keyframe_img, matches):
    query_2D = []
    keyframe_3D = []
    for match in matches:
        id_3D = keyframe_img.points2D[match.queryIdx].point3D_id
        if 0 <= id_3D < 18446744073709551615:
            query_2D.append(query_detector_kps[match.trainIdx].pt)
            a = reconstruction
            keyframe_3D.append(reconstruction.points3D[id_3D].xyz)
    return query_2D, keyframe_3D


if __name__ == '__main__':
    frameNames = os.listdir(images)

    # Assuming the frames are indexed
    frameNames.sort()

    frameNames = frameNames[:min(len(frameNames), 500)]

    # TODO: look into hloc and preprocess feature extraction and matching
    # retrieval_path = extract_features.main(retrieval_conf, images, image_list=frameNames, feature_path=features)
    # pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
    # feature_path = extract_features.main(feature_conf, images, outputs)
    # match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    # extract_features.main(feature_conf, images, image_list=frameNames, feature_path=features)
    # pairs_from_exhaustive.main(sfm_pairs, image_list=frameNames)
    # match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);

    # camera = pycolmap.infer_camera_from_image(images / frameNames[0])

    # Camera for Freiburg2/xyz
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
    min_tri_angle = 1.5

    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = max_angle_error
    options.continue_max_angle_error = max_angle_error
    options.merge_max_reproj_error = max_reproj_error
    options.complete_max_reproj_error = max_reproj_error

    # Triangulator that triangulates map points from 2D image correspondences
    triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction)

    # The chose feature detector and matcher
    used_extractor=enums.Extractors.ORB
    used_matcher = enums.Matchers.OrbHamming

    extractor,matcher=features.init(used_extractor,used_matcher)

    # Stores all the 3D map points that are currently in the reconstruction and its feature descriptor
    map_points = {}

    """ for SFM map init with hloc uncomment the lines
    references = frameNames[:20]
    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    model = hloc.reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
    model.add_camera(camera)
    # This overwrites the reconstruction using SFM map initialization
    reconstruction = model
    currFrameIdx = 20

    # Bc of pycolmap indexing start with 0
    frameNames.insert(0, "")
    success = True
    """

    success, currFrameIdx = map_initialization.initialize_map(images, frameNames, reconstruction,
                                                              graph,
                                                              triangulator, options, camera,
                                                              map_points,
                                                              extractor,matcher,
                                                              used_extractor,used_matcher)
    last_keyframeidx = currFrameIdx
    currFrameIdx += 1

    keyframe_idxes = []
    detector_map = {}
    # old_im = reconstruction.find_image_with_name(str(last_keyframeidx))

    f = open(str(outputs / "estimation.txt"), "w")
    references = []
    for c_img in reconstruction.images.values():
        keyframe_idxes.append(c_img.image_id)
        # extracting keypoints snd the relative descriptors of an image
        kp, detector_map[c_img.image_id] = features.detector(images,frameNames[c_img.image_id],extractor,used_extractor)

        f.write(img_to_name(frameNames[c_img.image_id], c_img))
        # references.append(frameNames[c_img.image_id])

    keyframe_idxes.sort()

    # fig = viz_3d.init_figure()
    # viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
    # viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(0,255,0,0.5)', name="map_init")
    # fig.show()

    while success and currFrameIdx < len(frameNames):
        old_im = reconstruction.images[last_keyframeidx]
        # https://vision.in.tum.de/data/datasets/rgbd-dataset/online_evaluation
        # For the evaluation use freiburg2/xyz and the estimation.txt from out/test1/sfm

        # extracting keypoints snd the relative descriptors of an image
        kp, detector_map[currFrameIdx] = features.detector(images ,frameNames[currFrameIdx],extractor,used_extractor)

        points2D = [keypoint.pt for keypoint in kp]
        # add image and correspondence to graph
        graph.add_image(currFrameIdx, len(points2D))

        q_pts = []
        k_pts = []
        last_keyframes = keyframe_idxes[-min(len(keyframe_idxes), 3):]

        # Goes over the last keyframes and checks all matches to make an estimation of the global camera pose
        for idx in last_keyframes:
            # Extracts all matches
            matches = features.matcher(detector_map[idx], detector_map[currFrameIdx],matcher,
                                                    used_matcher)
            # Functions that relates every keypoint in the image to a 3D point in the graph (if such a point exists)
            query_pts, keyframe_pts = match_to_3D_correspondences(kp,
                                                                  reconstruction.images[idx], matches)
            for (q_pt, k_pt) in zip(query_pts, keyframe_pts):
                if q_pt not in q_pts:
                    q_pts.append(q_pt)
                    k_pts.append(k_pt)

            # bring matches in the right form
            matches = [(match.queryIdx, match.trainIdx) for match in matches]
            matches = np.array(matches, dtype=np.uint32)
            # matches = matches[np.array(answer["inliers"], dtype=bool)]

            graph.add_correspondences(idx, currFrameIdx, matches)

        # Estimate absolute pose of the query image
        answer = pycolmap.absolute_pose_estimation(q_pts,
                                                   k_pts,
                                                   camera, max_error_px=12.0)

        if answer["success"]:
            # print("Frame", currFrameIdx, "sucess")
            im = pycolmap.Image(id=currFrameIdx, name=str(currFrameIdx), camera_id=camera.camera_id,
                                tvec=(answer["tvec"]), qvec=(answer["qvec"]))

            im.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
            im.registered = True
            reconstruction.add_image(im)

            # TODO should be done after motion only BA and for motuion only ba just use the inliers
            # Triangulate the points of the image based on the pose graph correlations and the 2D-3D correspondences
            num_tri = triangulator.triangulate_image(options, im.image_id)
            # print("triangulated", num_tri, " new 3D points")
            num_completed_obs = triangulator.complete_all_tracks(options)
            num_merged_obs = triangulator.merge_all_tracks(options)

            # ret_f = reconstruction.filter_points3D_in_images(max_reproj_error, min_tri_angle, {im.image_id})
            ret_f = reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle)
            # print("Filtered", ret_f, "3D points out")

            # Using optimization to correct the image pose:
            motion_ba = optimization.BundleAdjuster(reconstruction)
            # Set initial image pose as fixed
            motion_ba.constant_pose = [keyframe_idxes[0]]
            motion_ba.constant_tvec = [keyframe_idxes[1]]
            motion_ba.motion_only_BA([im.image_id])

            # num_trib = triangulator.triangulate_image(options, im.image_id)
            # print("triangulated", num_tri, " new 3D points")
            # num_completed_obs = triangulator.complete_all_tracks(options)
            # num_merged_obs = triangulator.merge_all_tracks(options)

            # ret_f = reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle)

            # E. New Keyframe Decision (See orb slam paper, missing 1) and 3) )
            if currFrameIdx - last_keyframeidx > 20 and reconstruction.images[currFrameIdx].num_points3D() < 0.9 * \
                    reconstruction.images[last_keyframeidx].num_points3D():
                last_keyframeidx = currFrameIdx
                keyframe_idxes.append(currFrameIdx)
                # For evaluation of dataset purposes
                f.write(img_to_name(frameNames[currFrameIdx], reconstruction.images[im.image_id]))
            else:
                reconstruction.deregister_image(im.image_id)
        else:
            print("Frame ", currFrameIdx, "failure: not able to estimate absolute pose")

        # Using global BA after a certain increase in the model
        if currFrameIdx % 250 == 0:
            global_ba = optimization.BundleAdjuster(reconstruction)
            global_ba.global_BA()
        currFrameIdx += 1

    # num_completed_obs = triangulator.complete_all_tracks(options)
    # triangulator.merge_all_tracks(options)
    # triangulator.retriangulate(options)
    # num_merged_obs = triangulator.merge_all_tracks(options)
    # print(num_completed_obs)
    # print(num_merged_obs)

    f.close()

    rec = pycolmap.Reconstruction()
    rec.add_camera(camera)
    for p in reconstruction.points3D.values():
        rec.add_point3D(p.xyz, pycolmap.Track(), np.zeros(3))
    for im in [img for img in reconstruction.images.values() if img.registered]:
        rec.add_image(im)
    reconstruction.export_PLY(exports)
    # rec.export_PLY(points_exports)

    if USE_OPEN_3D:
        pcd = o3d.io.read_point_cloud(str(exports))
        print(rec)
        o3d.visualization.draw_geometries([pcd])


    else:
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(255,0,0)')
        fig.show()
