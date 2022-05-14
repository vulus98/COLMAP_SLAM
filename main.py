import os
from pathlib import Path
import pycolmap
from src import enums, images_manager, incremental_mapper
from hloc.utils import viz_3d
from Utility.logger_setup import get_logger
import numpy as np

try:
    from src import viz
except ImportError as e:
    print('Failed to load open3d, defaulting to HLOC viz')
    viz = None

logger = get_logger(__name__)

images = Path('./data/rgbd_dataset_freiburg2_xyz/rgb/')
# images = Path('data/kitti/frames/')
outputs = Path('./out/test1/')
exports = outputs / 'reconstruction.ply'

# The growth rates after which to perform global bundle adjustment.\
ba_global_images_ratio = 1.1
ba_global_points_ratio = 1.1
ba_global_images_freq = 500
ba_global_points_freq = 250000

if __name__ == '__main__':
    frame_names = os.listdir(images)
    # Assuming the frames are indexed
    frame_names.sort()
    frame_names=frame_names[::20]
    frame_names = frame_names[:min(len(frame_names), 10)]


   # Camera for Freiburg2/xyz
    camera = pycolmap.Camera(
        model='PINHOLE',
        width=640,
        height=480,
        params=[525, 525, 319.5, 239.5],
    )

    # camera = pycolmap.Camera(
    #     model='PINHOLE',
    #     width=1280,
    #     height=1024,
    #     params=[1280*0.535719308086809,	1024*0.669566858850269,	1280*0.493248545285398,	1024*0.500408664348414],
    # )

    camera.camera_id = 0
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(camera)
    graph = pycolmap.CorrespondenceGraph()

    # The chose feature detector and matcher
    # used_extractor = enums.Extractors.ORB
    # used_matcher = enums.Matchers.OrbHamming

    used_extractor = enums.Extractors.SuperPoint
    used_matcher = enums.Matchers.SuperGlue

    # Number of images to compare a frame to
    init_max_num_images = 10

    # Adds all the images to the reconstruction and correspondence graph
    # Also used for matches and correspondence graph updates
    imgs_manager = images_manager.ImagesManager(images, frame_names, reconstruction, graph, camera, init_max_num_images,
                                                used_extractor, used_matcher)
    mapper = incremental_mapper.IncrementalMapper()
    mapper.BeginReconstruction(reconstruction, graph, imgs_manager)

    inc_mapper_options = incremental_mapper.IncrementalMapperOptions()
    inc_mapper_options.init_max_num_images = init_max_num_images
    # -1 means we try to initialize from the best pair we can find
    image_id1 = -1
    image_id2 = -1
    # Tries to find a good initial image pair
    success, image_id1, image_id2 = mapper.FindInitialImagePair(inc_mapper_options, image_id1, image_id2)
    if not success:
        logger.warning("No good initial image pair found")
        exit(1)
    reg_init_success = mapper.RegisterInitialImagePair(inc_mapper_options, image_id1, image_id2)
    if not reg_init_success:
        logger.warning("No registration for initial image pair")
        exit(1)

    if success:
        logger.info(f"Initializing map with image pair {image_id1} and {image_id2}")

    logger.info(f"Before bundle Adjustment: {mapper.reconstruction_.summary()}")

    """
    # ========= DEBUG
    rec = pycolmap.Reconstruction()
    rec.add_camera(camera)
    for p in mapper.reconstruction_.points3D.values():
        rec.add_point3D(p.xyz, pycolmap.Track(), np.zeros(3))
    registered_list = [img for img in mapper.reconstruction_.images.values() if img.registered]
    # every_nth_element = max(1, int(len(registered_list) / 30))
    for im in registered_list:  # [::every_nth_element]:
        rec.add_image(im)
    # ========= DEBUG
    """

    a = mapper.AdjustGlobalBundle(inc_mapper_options)
    b = mapper.FilterPoints(inc_mapper_options)
    # Most likely not needed
    c = mapper.FilterImages(inc_mapper_options)

    # for i in range(0,7):
    #     next_image_ids = mapper.FindNextImages(inc_mapper_options)
    #     mapper.RegisterNextImage(inc_mapper_options, next_image_ids[0])

    if reconstruction.num_reg_images() == 0 or reconstruction.num_points3D() == 0:
        print("To many points have been filtered out or image(s) not valid")
        exit(1)

    # Not final yet
    num_img_last_global_ba = 2
    num_points_last_global_ba = reconstruction.num_points3D()

    num_images = 2
    next_image_ids = mapper.FindNextImages(inc_mapper_options)
    while len(next_image_ids) > 0:
        i = 0
        while not mapper.RegisterNextImage(inc_mapper_options, next_image_ids[i]):
            i += 1
        num_images += 1
        if num_img_last_global_ba * ba_global_images_ratio < num_images \
                and abs(num_images - num_img_last_global_ba) < ba_global_images_ratio \
                and num_points_last_global_ba * ba_global_points_ratio < reconstruction.num_points3D() \
                and abs(reconstruction.num_points3D() - num_points_last_global_ba) < ba_global_points_freq:
            mapper.AdjustLocalBundle(inc_mapper_options, None, None, next_image_ids[i], mapper.GetModifiedPoints3D())
        else:
            mapper.AdjustGlobalBundle(inc_mapper_options)
            num_img_last_global_ba = num_images
            num_points_last_global_ba = reconstruction.num_points3D()
        next_image_ids = mapper.FindNextImages(inc_mapper_options)
    # mapper.EndReconstruction(False)
    # # TODO: implement something similar to: https://github.com/colmap/colmap/blob/e180948665b03c4a12d45e2ca39a589f42fdbda6/src/controllers/incremental_mapper.cc/#L379-L632

    rec = pycolmap.Reconstruction()
    rec.add_camera(camera)
    for p in mapper.reconstruction_.points3D.values():
        rec.add_point3D(p.xyz, pycolmap.Track(), np.zeros(3))
    registered_list = [img for img in mapper.reconstruction_.images.values() if img.registered]
    # every_nth_element = max(1, int(len(registered_list) / 30))
    for im in registered_list:  # [::every_nth_element]:
        rec.add_image(im)

    logger.info(f"After bundle Adjustment: {mapper.reconstruction_.summary()}")

    if viz:
        viz.show(reconstruction, str(images))
    else:
        fig = viz_3d.init_figure()

        # viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(0,255,0)')
        viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(255,255,255)')
        fig.show()

