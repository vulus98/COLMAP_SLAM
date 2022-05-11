import pycolmap
import numpy as np
from src.enums import ImageSelectionMethod
from Utility.logger_setup import get_logger

logger = get_logger(__name__)


# TODO: add keyframe list
# register next image, find next images
class IncrementalMapperOptions:

    # Minimum number of inliers for initial image pair.
    init_min_num_inliers = 100

    # Maximum error in pixels for two-view geometry estimation for initial
    # image pair.
    init_max_error = 4.0

    # Maximum forward motion for initial image pair.
    init_max_forward_motion = 0.95

    # Minimum triangulation angle for initial image pair.
    init_min_tri_angle = 16.0

    # Maximum number of trials to use an image for initialization.
    init_max_reg_trials = 2

    # Maximum reprojection error in absolute pose estimation.
    abs_pose_max_error = 12.0

    # Minimum number of inliers in absolute pose estimation.
    abs_pose_min_num_inliers = 30

    # Minimum inlier ratio in absolute pose estimation.
    abs_pose_min_inlier_ratio = 0.25

    # Whether to estimate the focal length in absolute pose estimation.
    abs_pose_refine_focal_length = True

    # Whether to estimate the extra parameters in absolute pose estimation.
    abs_pose_refine_extra_params = True

    # Number of images to optimize in local bundle adjustment.
    local_ba_num_images = 6

    # Minimum triangulation for images to be chosen in local bundle adjustment.
    local_ba_min_tri_angle = 6

    # Thresholds for bogus camera parameters.Images with bogus camera
    # parameters are filtered and ignored in triangulation.
    min_focal_length_ratio = 0.1  # Opening angle of ~130deg
    max_focal_length_ratio = 10  # Opening angle of ~5deg
    max_extra_param = 1

    # Maximum reprojection error in pixels for observations.
    filter_max_reproj_error = 4.0

    # Minimum triangulation angle in degrees for stable 3D points.
    filter_min_tri_angle = 1.5

    # Maximum number of trials to register an image.
    max_reg_trials = 3

    # If reconstruction is provided as input fix the existing image poses.
    fix_existing_images = False

    # Number of threads.
    num_threads = -1

    # Method to find and select next best image to register.
    image_selection_method = ImageSelectionMethod.MIN_UNCERTAINTY

    # Number of images to check for finding InitialImage pair
    init_max_num_images = 60

    # TODO add a check methods for parameters
    def check(self):
        return True


class IncrementalMapper:
    # =========================== "private" ===============================

        # Class that holds data of the reconstruction.
    reconstruction_ = None

    # Class that holds the correspondece graph
    graph_ = None

    # Class that is responsible for incremental triangulation.
    triangulator_ = None

    # The class that manages the images
    images_manager_ = None

    # Invalid imageId
    kInvalidImageId = -1

    # Number of images that are registered in at least on reconstruction.
    num_total_reg_images_ = 0

    # Number of shared images between current reconstruction and all other
    # previous reconstructions.
    num_shared_reg_images_ = 0

    # Estimated two-view geometry of last call to `FindFirstInitialImage`,
    # used as a cache for a subsequent call to `RegisterInitialImagePair`.
    prev_init_image_pair_id_ = (-1, -1)
    prev_init_two_view_geometry_ = None

    # Images and image pairs that have been used for initialization. Each image
    # and image pair is only tried once for initialization.
    init_num_reg_trials_ = {}
    init_image_pairs_ = []

    # The number of registered images per camera. This information is used
    # to avoid duplicate refinement of camera parameters and degradation of
    # already refined camera parameters in local bundle adjustment when multiple
    # images share intrinsics.
    num_reg_images_per_camera_ = {}

    # The number of reconstructions in which images are registered.
    num_registrations_ = {}

    # Images that have been filtered in current reconstruction.
    filtered_images_ = None

    # Number of trials to register image in current reconstruction. Used to set
    # an upper bound to the number of trials to register an image.
    num_reg_trials_ = {}

    # Images that were registered before beginning the reconstruction.
    # This image list will be non-empty, if the reconstruction is continued from
    # an existing reconstruction.
    existing_image_ids_ = None

    #TODO: Add keyframe store

    # This function does not seem to get exposed from pycolmap
    def DegToRad(self, deg):
        return deg * np.pi / 180

    # Find seed images for incremental reconstruction. Suitable seed images have
    # a large number of correspondences and have camera calibration priors. The
    # returned list is ordered such that most suitable images are in the front.
    def FindFirstInitialImage(self, options):

        init_max_reg_trials = options.init_max_reg_trials

        # Collect information of all not yet registered images with
        # correspondences. We consider only the options.
        image_infos = []

        max_len = min(self.reconstruction_.num_images(), options.init_max_num_images)
        for image in [self.reconstruction_.images[img_id] for img_id in self.images_manager_.image_ids[:max_len]]:
            # Only images with correspondences can be registered.
            if image.num_correspondences == 0:
                continue

            # Only use images for initialization a maximum number of times.
            if self.init_num_reg_trials_.get(image.image_id, 0) >= options.init_max_reg_trials:
                continue

            # Only use images for initialization that are not registered in any
            # of the other reconstructions.
            if self.num_registrations_.get(image.image_id, 0) > 0:
                continue

            camera = self.reconstruction_.cameras[image.camera_id]
            image_info = {
                "image_id": image.image_id,
                "prior_focal_length": camera.has_prior_focal_length,
                "num_correspondences": image.num_correspondences
            }
            image_infos.append(image_info)

        # Sort images such that images with a prior focal length and more
        # correspondences are preferred, i.e. they appear in the front of the list.
        image_infos = sorted(image_infos, key=lambda d: (not d["prior_focal_length"], d["num_correspondences"]), reverse=True)

        # Extract image identifiers in sorted order.
        image_ids = [image_info["image_id"] for image_info in image_infos]

        return image_ids

    # For a given first seed image, find other images that are connected to the
    # first image. Suitable second images have a large number of correspondences
    # to the first image and have camera calibration priors. The returned list is
    # ordered such that most suitable images are in the front.
    def FindSecondInitialImage(self, options, image_id1):
        image1 = self.reconstruction_.images[image_id1]
        num_correspondences = {}
        for point2D_idx in range(image1.num_points2D()):
            for corr in self.graph_.find_correspondences(image_id1, point2D_idx):
                if self.num_registrations_.get(corr.image_id, 0) == 0:
                    num_correspondences[corr.image_id] = num_correspondences.get(corr.image_id, 0) + 1

        init_min_num_inliers = options.init_min_num_inliers
        image_infos = []
        for k, v in num_correspondences.items():
            if v >= init_min_num_inliers:
                image = self.reconstruction_.images[k]
                camera = self.reconstruction_.cameras[image.camera_id]
                image_info = {
                    "image_id": k,
                    "prior_focal_length": camera.has_prior_focal_length,
                    "num_correspondences": v
                }
                image_infos.append(image_info)

        # Sort images such that images with a prior focal length and more
        # correspondences are preferred, i.e. they appear in the front of the list.
        image_infos = sorted(image_infos, key=lambda d: (not d["prior_focal_length"], d["num_correspondences"]), reverse=True)

        # Extract image identifiers in sorted order.
        image_ids = [image_info["image_id"] for image_info in image_infos]

        return image_ids


    # Find local bundle for given image in the reconstruction. The local bundle
    # is defined as the images that are most connected, i.e. maximum number of
    # shared 3D points, to the given image.
    def FindLocalBundle(self, options, image_id):
        a = 0

    # Register / De-register image in current reconstruction and update
    # the number of shared images between all reconstructions.
    def RegisterImageEvent(self, image_id):
        image = self.reconstruction_.images[image_id]
        num_reg_images_for_camera = self.num_reg_images_per_camera_.get(image.camera_id, 0)

        num_reg_images_for_camera += 1

        num_regs_for_image = self.num_registrations_.get(image_id, 0)
        num_regs_for_image += 1
        if num_regs_for_image == 1:
            self.num_total_reg_images_ += 1
        elif num_regs_for_image > 1:
            self.num_shared_reg_images_ += 1

    def DeRegisterImageEvent(self, image_id):
        image = self.reconstruction_.images[image_id]
        num_reg_images_for_camera = self.num_reg_images_per_camera_.get(image.CameraId(), 0)

        if num_reg_images_for_camera > 0:
            num_reg_images_for_camera -= 1

            num_regs_for_image = self.num_registrations_.get(image_id, 0)
            num_regs_for_image -= 1
            if num_regs_for_image == 0:
                self.num_total_reg_images_ -= 1
            elif num_regs_for_image > 0:
                self.num_shared_reg_images_ -= 1

    def EstimateInitialTwoViewGeometry(self, options, image_id1, image_id2):
        image_pair_id = self.images_manager_.ImagePairToPairId(image_id1, image_id2)

        if self.prev_init_image_pair_id_ == image_pair_id:
            return True

        # TODO: this should get loaded from the database, make sure this way works to
        image1 = self.reconstruction_.images[image_id1]
        camera1 = self.reconstruction_.cameras[image1.camera_id]

        image2 = self.reconstruction_.images[image_id2]
        camera2 = self.reconstruction_.cameras[image2.camera_id]

        matches = self.graph_.find_correspondences_between_images(image_id1, image_id2)

        points1 = [image1.points2D[point_id[0]].xy for point_id in matches]

        points2 = [image2.points2D[point_id[1]].xy for point_id in matches]

        two_view_geometry_options = pycolmap.TwoViewGeometryOptions()
        two_view_geometry_options.ransac.min_num_trials = 30
        two_view_geometry_options.ransac.max_error = options.init_max_error

        answer = pycolmap.two_view_geometry_estimation(points1, points2, camera1, camera2, two_view_geometry_options)

        if not answer["success"]:
            return False

        # Some optical flow/motion constraint to make sure the baseline is big enough
        flow_constr = sum(abs(np.array(points1) - np.array(points2))[answer["inliers"]])
        flow_constr = flow_constr / sum(answer["inliers"])
        # Just using the one camera assuming we have only one
        fx = camera1.focal_length_x
        fy = camera1.focal_length_y
        f = np.array([fx, fy])
        flow_constr = sum(flow_constr / f)

        if flow_constr > 0.09 and answer["num_inliers"] >= options.init_min_num_inliers and abs(answer["tvec"][2]) < options.init_max_forward_motion:
            # TODO: Note the Colmap code checks also the triagulation angle but this seems not really possible with the pycolmap bindings
            # see: https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc/#L216-L217
            self.prev_init_image_pair_id_ = image_pair_id
            self.prev_init_two_view_geometry_ = answer
            return True

        return False
    # ========================= "public" ======================================

    # Create incremental mapper.
    def __init__(self):
        self.reconstruction_ = None
        self.num_total_reg_images_ = 0
        self.num_shared_reg_images_ = 0
        self.prev_init_image_pair_id_ = (-1, -1)

    # Prepare the mapper for a new reconstruction which is empty
    # (in which case `RegisterInitialImagePair` must be called).
    def BeginReconstruction(self, reconstruction, graph, images_manager):
        if self.reconstruction_ != None:
            print("Reconstruction objec;t in Incremental Mapper should be empty!")
        self.reconstruction_ = reconstruction
        self.graph_ = graph
        self.images_manager_ = images_manager
        self.triangulator_ = pycolmap.IncrementalTriangulator(self.graph_, self.reconstruction_)

        self.num_shared_reg_images_ = 0
        self.num_reg_images_per_camera_ = {}

        for img_id in reconstruction.reg_image_ids():
            self.RegisterImageEvent(img_id)

        self.existing_image_ids_ = reconstruction.reg_image_ids()

        self.prev_init_image_pair_id_ = (-1, -1)
        self.prev_init_two_view_geometry_ = None

        self.filtered_images_ = []
        self.num_reg_trials_ = {}

    # Cleanup the mapper after the current reconstruction is done. If the
    # model is discarded, the number of total and shared registered images will
    # be updated accordingly. Bool
    def EndReconstruction(self, discard):
        if self.reconstruction_ is None:
            print("Calling EndReconstuction on an empty reconstruction!")
        else:
            if discard:
                for img_ids in self.reconstruction_.reg_image_ids():
                    self.DeRegisterImageEvent(img_ids)

            self.reconstruction_ = None
            self.triangulator_ = None

    # Find initial image pair to seed the incremental reconstruction. The image
    # pairs should be passed to `RegisterInitialImagePair`. This function
    # automatically ignores image pairs that failed to register previously.
    def FindInitialImagePair(self, options, image_id1, image_id2):
        options.check()

        image_ids1 = []
        if image_id1 != self.kInvalidImageId and image_id2 == self.kInvalidImageId:
            # Only first image provided
            if self.images_manager_.exists_image(image_id1):
                return False, -1, -1

            image_ids1.append(image_id1)
        elif image_id1 == self.kInvalidImageId and image_id2 != self.kInvalidImageId:
            # Only second image provided
            if self.images_manager_.exists_image(image_id2):
                return False, -1, -1

            image_ids1.push_back(image_id2)
        else:
            image_ids1 = self.FindFirstInitialImage(options)

        # Try to find good initial pair:
        for i1 in range(len(image_ids1)):
            image_id1 = image_ids1[i1]

            image_ids2 = self.FindSecondInitialImage(options, image_id1)

            for i2 in range(len(image_ids2)):
                image_id2 = image_ids2[i2]
                pair_id = self.images_manager_.ImagePairToPairId(image_id1, image_id2)

                if self.init_image_pairs_.count(pair_id) > 0:
                    continue

                self.init_image_pairs_.append(pair_id)

                if self.EstimateInitialTwoViewGeometry(options, image_id1, image_id2):
                    return True, image_id1, image_id2

        return False, -1, -1

    # Find best next image to register in the incremental reconstruction. The
    # images should be passed to `RegisterNextImage`. This function automatically
    # ignores images that failed to registered for `max_reg_trials`.
    def FindNextKeyframe(self, options):
        print(options)
        print(f"Printing reconstruction summary from inside the findnextkeyframe function in incremental_mapper.py: {self.reconstruction_.summary()}")
        a = 0



        # condition1 = True
        # condition2 = currFrameIdx - last_keyframeidx > 20
        # condition3 = True
        # condition4 = self.reconstruction_.images[currFrameIdx].num_points3D() < 0.9 * \
        #         reconstruction.images[last_keyframeidx].num_points3D()
        # condition5 = True
        #
        # all_conditions_satisfied = np.all([condition1, condition2, condition3, condition4, condition5])
        #
        #
        # if all_conditions_satisfied:
        #
        #     self.triangulator_.complete_image(options, im.image_id)
        #     last_keyframeidx = currFrameIdx
        #     keyframe_idxes.append(currFrameIdx)
        #     # For evaluation of dataset purposes
        #     f.write(img_to_name(frameNames[currFrameIdx], reconstruction.images[im.image_id]))
        # else:
        #     reconstruction.deregister_image(im.image_id)



    # Attempt to seed the reconstruction from an image pair.
    def RegisterInitialImagePair(self, options, image_id1, image_id2):
        if self.reconstruction_ is None:
            print("Incremental Mapper (RegisterInitialImagePair): reconstruction is NONE!")

        if self.reconstruction_.num_reg_images() != 0:
            print("Incremental Mapper (RegisterInitialImagePair): reconstruction has already images registered!")

        options.check()

        self.init_num_reg_trials_[image_id1] = self.init_num_reg_trials_.get(image_id1, 0) + 1
        self.init_num_reg_trials_[image_id2] = self.init_num_reg_trials_.get(image_id2, 0) + 1
        self.num_reg_trials_[image_id1] = self.num_reg_trials_.get(image_id1, 0) + 1
        self.num_reg_trials_[image_id2] = self.num_reg_trials_.get(image_id2, 0) + 1

        pair_id = self.images_manager_.ImagePairToPairId(image_id1, image_id2)
        self.init_image_pairs_.append(pair_id)

        image1 = self.reconstruction_.images[image_id1]
        camera1 = self.reconstruction_.cameras[image1.camera_id]

        image2 = self.reconstruction_.images[image_id2]
        camera2 = self.reconstruction_.cameras[image2.camera_id]

        # ==========================
        # Estimate two-view geometry
        # ==========================

        if not self.EstimateInitialTwoViewGeometry(options, image_id1, image_id2):
            return False

        R = np.eye(3)
        qv = pycolmap.rotmat_to_qvec(R)
        tv = [0, 0, 0]

        image1.qvec = qv
        image1.tvec = tv
        image2.qvec = self.prev_init_two_view_geometry_["qvec"]
        image2.tvec = self.prev_init_two_view_geometry_["tvec"]

        proj_matrix1 = image1.projection_matrix()
        proj_matrix2 = image2.projection_matrix()
        proj_center1 = image1.projection_center()
        proj_center2 = image2.projection_center()

        # ==========================
        # Update Reconstruction
        # ==========================

        self.reconstruction_.register_image(image_id1)
        self.reconstruction_.register_image(image_id2)
        self.RegisterImageEvent(image_id1)
        self.RegisterImageEvent(image_id2)

        # corrs = self.graph_.find_correspondences_between_images(image_id1, image_id2)

        # TODO check that triangulation alone works and we do not have to manually set these things
        '''
        # Add 3D point tracks
        track = pycolmap.Track()
        track.add_element(pycolmap.TrackElement())
        track.add_element(pycolmap.TrackElement())
        track.elements[0].image_id = image_id1
        track.elements[1].image_id = image_id2
        for corr in corrs:
            point1_N = camera1.image_to_world(image1.points2D[corr.point2D_idx1].xy)
            point2_N = camera2.image_to_world(image2.points2D[corr.point2D_idx2].xy)
        ...
        '''

        min_tri_angle_rad = self.DegToRad(options.init_min_tri_angle)
        triang_options = pycolmap.IncrementalTriangulatorOptions()
        triang_options.ignore_two_view_track = False
        self.triangulator_.triangulate_image(triang_options, image_id1)
        # Filter3D points with large reprojection error, negative depth, or
        # insufficient triangulation angle
        self.reconstruction_.filter_all_points3D(options.init_max_error, min_tri_angle_rad)

        return True


    # Attempt to register image to the existing model. This requires that
    # a previous call to `RegisterInitialImagePair` was successful.
    def RegisterNextImage(self, options, image_id):
        a = 0

    # Triangulate observations of image.
    def TriangulateImage(self, tri_options, image_id):
        a = 0

    # Retriangulate image pairs that should have common observations according to
    # the scene graph but don't due to drift, etc. To handle drift, the employed
    # reprojection error thresholds should be relatively large. If the thresholds
    # are too large, non-robust bundle adjustment will break down: if the
    # thresholds are too small, we cannot fix drift effectively.
    def Retriangulate(self, tri_options):
        a = 0

    # Complete tracks by transitively following the scene graph correspondences.
    # This is especially effective after bundle adjustment, since many cameras
    # and point locations might have improved. Completion of tracks enables
    # better subsequent registration of new images.
    def CompleteTracks(self, tri_options):
        a = 0

    # Merge tracks by using scene graph correspondences. Similar to
    # `CompleteTracks`, this is effective after bundle adjustment and improves
    # the redundancy in subsequent bundle adjustments.
    def MergeTracks(self, tri_options):
        a = 0

    # Adjust locally connected images and points of a reference image. In
    # addition, refine the provided 3D points. Only images connected to the
    # reference image are optimized. If the provided 3D points are not locally
    # connected to the reference image, their observing images are set as
    # constant in the adjustment.
    # options: IncrementalMapper Options
    # ba_options: Bundle adjustment options
    # tri_options: Triangulator options
    def AdjustLocalBundle(self, options, ba_options, tri_options, image_id, point3D_ids):
        a = 0

    # Global bundle adjustment using Ceres Solver or PBA.
    def AdjustGlobalBundle(self, options, ba_options):
        a = 0

    # Filter images and point observations.
    def FilterImages(self, options):
        a = 0

    def FilterPoints(self, options):
        a = 0

    def GetReconstruction(self):
        a = 0

    # Number of images that are registered in at least on reconstruction.
    def NumTotalRegImages(self):
        a = 0

    # Number of shared images between current reconstruction and all other
    # previous reconstructions.
    def NumSharedRegImages(self):
        a = 0

    # Get changed 3D points, since the last call to `ClearModifiedPoints3D`.
    def GetModifiedPoints3D(self):
        a = 0

    # Clear the collection of changed 3D points.
    def ClearModifiedPoints3D(self):
        a = 0
