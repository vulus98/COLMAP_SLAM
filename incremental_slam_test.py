import os
from pathlib import Path
import pycolmap
import pyceres
from src import enums, images_manager, incremental_mapper, features
from hloc.utils import viz_3d
import numpy as np

# images = Path('data/frames/test1/')
images = Path('data/rgbd_dataset_freiburg2_xyz/rgb/')
outputs = Path('out/test1/')
exports = outputs / 'reconstruction.ply'

if __name__ == '__main__':
    frame_names = os.listdir(images)

    # Assuming the frames are indexed
    frame_names.sort()

    frame_names = frame_names[:min(len(frame_names), 200)]

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

    # The chose feature detector and matcher
    # used_extractor = enums.Extractors.SuperPoint
    # used_matcher = enums.Matchers.SuperGlue

    used_extractor = enums.Extractors.ORB
    used_matcher = enums.Matchers.OrbHamming

    # Number of images to comapare a frame to
    init_max_num_images = 60

    # Adds all the images to the reconstruction and correspondence graph
    # Also used for matches and correspondence graph updates
    imgs_manager = images_manager.ImagesManager(images, frame_names, reconstruction, graph, camera, init_max_num_images, used_extractor, used_matcher)
    mapper = incremental_mapper.IncrementalMapper()
    mapper.BeginReconstruction(reconstruction, graph, imgs_manager)

    inc_mapper_options = incremental_mapper.IncrementalMapperOptions()
    inc_mapper_options.init_max_num_images = init_max_num_images
    # Tries to find a good initial image pair
    sucess, image_id1, image_id2 = mapper.FindInitialImagePair(inc_mapper_options, -1, -1)
    if not sucess:
        print("No good initial image pair found")
        exit(-1)
    if not mapper.RegisterInitialImagePair(inc_mapper_options, image_id1, image_id2):
        print("No registration for initial image pair")
        exit(-1)

    print("Initialized with images:", image_id1, "and", image_id2)
    bundle_adj_options = pyceres.SolverOptions()
    # a = mapper.AdjustGlobalBundle(inc_mapper_options, bundle_adj_options)
    b = mapper.FilterPoints(inc_mapper_options)
    # Most likely not needed
    # c = mapper.FilterImages(inc_mapper_options)

    if reconstruction.num_reg_images() == 0 or reconstruction.num_points3D() == 0:
        print("To many points have been filtered out or image(s) not valid")
        exit(-1)

    rec = pycolmap.Reconstruction()
    rec.add_camera(camera)
    for p in mapper.reconstruction_.points3D.values():
        rec.add_point3D(p.xyz, pycolmap.Track(), np.zeros(3))
    registered_list = [img for img in mapper.reconstruction_.images.values() if img.registered]
    # every_nth_element = max(1, int(len(registered_list) / 30))
    for im in registered_list:# [::every_nth_element]:
        rec.add_image(im)

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(255,0,0)')
    fig.show()

    # Example usage:
    # while (...) {
    #   const auto next_image_ids = mapper.FindNextImages(options);
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