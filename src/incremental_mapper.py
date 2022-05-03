import pycolmap
from enums import ImageSelectionMethod


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

    # TODO add a check methods for parameters

class IncrementalMapper:
    # =========================== "private" ===============================

    # Find seed images for incremental reconstruction. Suitable seed images have
    # a large number of correspondences and have camera calibration priors. The
    # returned list is ordered such that most suitable images are in the front.
    def FindFirstInitialImage(self, options):
        a = 0

    # For a given first seed image, find other images that are connected to the
    # first image. Suitable second images have a large number of correspondences
    # to the first image and have camera calibration priors. The returned list is
    # ordered such that most suitable images are in the front.
    def FindSecondInitialImage(self, options, image_id1):
        a = 0

    # Find local bundle for given image in the reconstruction. The local bundle
    # is defined as the images that are most connected, i.e. maximum number of
    # shared 3D points, to the given image.
    def FindLocalBundle(self, options, image_id):
        a = 0

    # Register / De-register image in current reconstruction and update
    # the number of shared images between all reconstructions.
    def RegisterImageEvent(self, image_id):
        a = 0
    def DeRegisterImageEvent(self, image_id):
        a = 0

    def EstimateInitialTwoViewGeometry(self, options, image_id1, image_id2):
        a = 0

    # Class that holds data of the reconstruction.
    reconstruction_ = None

    # Class that is responsible for incremental triangulation.
    triangulator_ = None

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
    init_num_reg_trials_ = None
    init_image_pairs_ = None

    # The number of registered images per camera. This information is used
    # to avoid duplicate refinement of camera parameters and degradation of
    # already refined camera parameters in local bundle adjustment when multiple
    # images share intrinsics.
    num_reg_images_per_camera_ = None

    # The number of reconstructions in which images are registered.
    num_registrations_ = None

    # Images that have been filtered in current reconstruction.
    filtered_images_ = None

    # Number of trials to register image in current reconstruction. Used to set
    # an upper bound to the number of trials to register an image.
    num_reg_trials_ = 0

    # Images that were registered before beginning the reconstruction.
    # This image list will be non-empty, if the reconstruction is continued from
    # an existing reconstruction.
    existing_image_ids_ = None

    # ========================= "public" ======================================

    
    # Create incremental mapper. The database cache must live for the entire
    # life-time of the incremental mapper.
    def __init__(self):
        a = 0

    # Prepare the mapper for a new reconstruction, which might have existing
    # registered images (in which case `RegisterNextImage` must be called) or
    # which is empty (in which case `RegisterInitialImagePair` must be called).
    def BeginReconstruction(self, reconstruction):
        a = 0
    
    # Cleanup the mapper after the current reconstruction is done. If the
    # model is discarded, the number of total and shared registered images will
    # be updated accordingly. Bool
    def EndReconstruction(self, discard):
        a = 0
    
    # Find initial image pair to seed the incremental reconstruction. The image
    # pairs should be passed to `RegisterInitialImagePair`. This function
    # automatically ignores image pairs that failed to register previously.
    def FindInitialImagePair(self, options, image_id1, image_id2):
        a = 0
    
    # Find best next image to register in the incremental reconstruction. The
    # images should be passed to `RegisterNextImage`. This function automatically
    # ignores images that failed to registered for `max_reg_trials`.
    def FindNextImages(self, options):
        a = 0
    
    # Attempt to seed the reconstruction from an image pair.
    def RegisterInitialImagePair(self, options, image_id1, image_id2):
        a = 0
    
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