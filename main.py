import os
from pathlib import Path
import pycolmap
from src import enums, images_manager, incremental_mapper
from hloc.utils import viz_3d
from Utility.logger_setup import get_logger
import numpy as np

logger = get_logger(__name__)

images = Path('data/rgbd_dataset_freiburg1_xyz/rgb/')
# images = Path('data/kitti/frames/')
outputs = Path('out/test1/')
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

    frame_names = frame_names[:min(len(frame_names), 30)]

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

    # The chosen feature detector and matcher
    # used_extractor = enums.Extractors.SuperPoint
    # used_matcher = enums.Matchers.SuperGlue

    used_extractor = enums.Extractors.ORB
    used_matcher = enums.Matchers.OrbHamming

    # Number of images to compare a frame to
    init_max_num_images = 60

    # Adds all the images to the reconstruction and correspondence graph
    # Also used for matches and correspondence graph updates
    imgs_manager = images_manager.ImagesManager(images, frame_names, reconstruction, graph, camera, init_max_num_images, used_extractor, used_matcher)
    mapper = incremental_mapper.IncrementalMapper()
    mapper.BeginReconstruction(reconstruction, graph, imgs_manager)

    inc_mapper_options = incremental_mapper.IncrementalMapperOptions()
    inc_mapper_options.init_max_num_images = init_max_num_images
    # Tries to find a good initial image pair
    success, image_id1, image_id2 = mapper.FindInitialImagePair(inc_mapper_options, -1, -1)
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

    if reconstruction.num_reg_images() == 0 or reconstruction.num_points3D() == 0:
        print("To many points have been filtered out or image(s) not valid")
        exit(1)

    # Not final yet
    num_img_last_global_ba = 2
    num_points_last_global_ba = reconstruction.num_points3D()

    num_images = 2
    for i in range(max(image_id1, image_id2), len(images)):
        next_image_ids = mapper.FindNextImages()
        for image_id in next_image_ids:
            if mapper.RegisterNextImage(inc_mapper_options, image_id):
                num_images += 1
            if num_img_last_global_ba * ba_global_images_ratio < num_images \
                    and abs(num_images - num_img_last_global_ba) < ba_global_images_ratio \
                    and num_points_last_global_ba * ba_global_points_ratio < reconstruction.num_points3D() \
                    and abs(reconstruction.num_points3D() - num_points_last_global_ba) < ba_global_points_freq:
                mapper.AdjustLocalBundle(inc_mapper_options, None, None, image_id, None)
            else:
                mapper.AdjustGlobalBundle(inc_mapper_options)
                num_img_last_global_ba = num_images
                num_points_last_global_ba = reconstruction.num_points3D()
    mapper.EndReconstruction(False)
    # TODO: implement something similar to: https://github.com/colmap/colmap/blob/e180948665b03c4a12d45e2ca39a589f42fdbda6/src/controllers/incremental_mapper.cc/#L379-L632

    rec = pycolmap.Reconstruction()
    rec.add_camera(camera)
    for p in mapper.reconstruction_.points3D.values():
        rec.add_point3D(p.xyz, pycolmap.Track(), np.zeros(3))
    registered_list = [img for img in mapper.reconstruction_.images.values() if img.registered]
    # every_nth_element = max(1, int(len(registered_list) / 30))
    for im in registered_list:# [::every_nth_element]:
        rec.add_image(im)


    logger.info(f"After bundle Adjustment: {mapper.reconstruction_.summary()}")

    fig = viz_3d.init_figure()

    # viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(0,255,0)')
    viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(255,255,255)')
    fig.show()
