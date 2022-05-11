import os
from pathlib import Path
import pycolmap
from src import enums, images_manager, incremental_mapper
from hloc.utils import viz_3d
from Utility.logger_setup import get_logger

logger = get_logger(__name__)

images = Path('data/rgbd_dataset_freiburg2_xyz/rgb/')
images = Path('data/kitti/frames/')
outputs = Path('out/test1/')
exports = outputs / 'reconstruction.ply'

if __name__ == '__main__':
    frame_names = os.listdir(images)

    # Assuming the frames are indexed
    frame_names.sort()

    frame_names = frame_names[:min(len(frame_names), 3)] # TODO: reduced to 3 only for testing

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
        exit(-1)
    reg_init_success = mapper.RegisterInitialImagePair(inc_mapper_options, image_id1, image_id2)
    if not reg_init_success:
        print("No registration for initial image pair")
        exit(-1)

    if success:
        print(f"Initializing map with image pair {image_id1} and {image_id2}")

    find_next_keyframe = mapper.FindNextKeyframe(inc_mapper_options)

    print(f"Summary of the reconstruction at the end of the mapper: {mapper.reconstruction_.summary()}")

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, mapper.reconstruction_, min_track_length=0, color='rgb(255,0,0)')
    fig.show()

    # Adjust global bundle
    # filter points
    # filter images
    # print(reconstruction summary)

    # Example usage:
    # while (...) {
    #   const auto next_image_ids = mapper.FindNextKeyframe(options);
    #   for (const auto image_id: next_image_ids) {
    #       CHECK(mapper.RegisterNextImage(options, image_id));
    #       if (...) {
    #           mapper.AdjustLocalBundle(...);
#           } else {
    #           mapper.AdjustGlobalBundle(...);
    #       }
    #   }
    #}
    # mapper.EndReconstruction(false);
    # TODO: implement something similar to: https://github.com/colmap/colmap/blob/e180948665b03c4a12d45e2ca39a589f42fdbda6/src/controllers/incremental_mapper.cc/#L379-L632