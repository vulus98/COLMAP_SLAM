import os
from pathlib import Path
from cv2 import exp
import pycolmap
from src import enums, images_manager, incremental_mapper
from Utility.logger_setup import get_logger

logger = get_logger('pipeline.py')


class Pipeline:

    def __init__(self,
                 extractor=enums.Extractors(1),
                 matcher=enums.Matchers(1),
                 selector=enums.ImageSelectionMethod(1)):

        # Default camera
        self.camera = pycolmap.Camera(
            model='PINHOLE',
            width=640,
            height=480,
            params=[525, 525, 319.5, 239.5],
        )
        self.camera.camera_id = 0

        # pycolmap properties
        self.reset()
        self.mapper = incremental_mapper.IncrementalMapper()
        self.inc_mapper_options = incremental_mapper.IncrementalMapperOptions()

        # Bundle adjustment properties
        self.ba_global_images_ratio = 1.1
        self.ba_global_points_ratio = 1.1
        self.ba_global_images_freq = 500
        self.ba_global_points_freq = 250000

        self.extractor = extractor
        self.matcher = matcher
        self.selector = selector

        # Empty paths to be set during load
        self.image_path = ""
        self.output_path = ""
        self.export_name = ""

    def reset(self):
        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.add_camera(self.camera)
        self.graph = pycolmap.CorrespondenceGraph()

    def set_ba_properties(self, image_ratio=None, point_ratio=None, image_freq=None, point_freq=None):
        if image_ratio:
            self.ba_global_images_ratio = image_ratio

        if point_ratio:
            self.ba_global_points_ratio = point_ratio

        if image_freq:
            self.ba_global_images_freq = image_freq

        if point_freq:
            self.ba_global_points_freq = point_freq

    def load_data(self, images=None, outputs=None, exports=None, init_max_num_images=60, frame_skip=20, max_frame=10):
        if images:
            self.image_path = Path(images)

        if outputs:
            self.output_path = Path(outputs)

        if exports:
            if not self.output_path:
                logger.warning("Set output path first!")
            self.export_name = self.output_path / exports

        if self.image_path:
            self.frame_names = os.listdir(self.image_path)
            self.frame_names.sort()
            # Do not skip any frmaes if set to negative
            if frame_skip > 0:
                self.frame_names = self.frame_names[::frame_skip]
            # Use all available frames if negative
            if max_frame == -1:
                max_frame = len(self.frame_names)
            self.frame_names = self.frame_names[:min(len(self.frame_names), max_frame)]

        self.img_manager = images_manager.ImagesManager(self.image_path,
                                                        self.frame_names,
                                                        self.reconstruction,
                                                        self.graph,
                                                        self.camera,
                                                        init_max_num_images,
                                                        self.extractor,
                                                        self.matcher)

        if not self.image_path:
            logger.warning("NEED TO LOAD IMAGE DATA!")

        self.inc_mapper_options.init_max_num_images = init_max_num_images

    def run(self, image_id1=-1, image_id2=-1, init_max_trials=10):
        if not self.img_manager:
            logger.error("Load images first!")
            return

        self.mapper.BeginReconstruction(self.reconstruction, self.graph, self.img_manager)

        for i in range(init_max_trials):
            # Tries to find a good initial image pair
            success, image_id1, image_id2 = self.mapper.FindInitialImagePair(self.inc_mapper_options, image_id1, image_id2)
            if not success:
                logger.warning("No good initial image pair found")
                exit(1)
            reg_init_success = self.mapper.RegisterInitialImagePair(self.inc_mapper_options, image_id1, image_id2)
            if not reg_init_success:
                logger.warning("No registration for initial image pair")
                exit(1)

            if success:
                logger.info(f"Initializing map with image pair {image_id1} and {image_id2}")

            logger.info(f"Before bundle Adjustment: {self.mapper.reconstruction_.summary()}")

            if self.reconstruction.num_reg_images() == 0 or self.reconstruction.num_points3D() < 150:
                print("Not enough 3D points found after triangulation!")
                self.mapper.ClearReconstruction()
            else:
                self.mapper.AdjustGlobalBundle(self.inc_mapper_options)
                self.mapper.FilterPoints(self.inc_mapper_options)
                self.mapper.FilterImages(self.inc_mapper_options)

                if self.reconstruction.num_reg_images() == 0 or self.reconstruction.num_points3D() < 130:
                    print("To many points have been filtered out or image(s) not valid")
                    self.mapper.ClearReconstruction()
                else:
                    break

        logger.info(f"After Map initialization: {self.mapper.reconstruction_.summary()}")

        # Not final yet
        num_img_last_global_ba = 2
        num_points_last_global_ba = self.reconstruction.num_points3D()
        print(self.reconstruction.num_points3D())

        num_images = 2

        success_register_keyframe = True
        iteration_count = 1

        while success_register_keyframe:

            # Iterate through all images until you hit a keyframe and successfully register it.
            keyframe_id, success_register_keyframe = self.mapper.FindAndRegisterNextKeyframe(self.inc_mapper_options)

            # if not successful, all images have been processed, and this while loop will terminate
            if not success_register_keyframe:
                continue

            # Bundle Adjustment
            num_images += 1
            if num_img_last_global_ba * self.ba_global_images_ratio < num_images \
                    and abs(num_images - num_img_last_global_ba) < self.ba_global_images_ratio \
                    and num_points_last_global_ba * self.ba_global_points_ratio < self.reconstruction.num_points3D() \
                    and abs(self.reconstruction.num_points3D() - num_points_last_global_ba) < self.ba_global_points_freq:
                self.mapper.AdjustLocalBundle(self.inc_mapper_options, None, None, keyframe_id, None)
            else:
                self.mapper.AdjustGlobalBundle(self.inc_mapper_options)
                num_img_last_global_ba = num_images
                num_points_last_global_ba = self.reconstruction.num_points3D()

            logger.info(f"Iteration {iteration_count} of keyframe selection: {self.reconstruction.summary()}")
            iteration_count += 1
        logger.info(f"Final: {self.mapper.reconstruction_.summary()}")

    def vizualize(self, vizualizer='hloc'): # or vizualizer='open3d'
        # logger.info(f"After bundle Adjustment: {self.mapper.reconstruction_.summary()}")

        if vizualizer == 'hloc':
            from hloc.utils import viz_3d
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig, self.reconstruction, min_track_length=0, color='rgb(255,0,0)')
            fig.show()

        elif vizualizer == 'open3d':
            try:
                from src import viz
            except ImportError:
                logger.warning('Failed to load open3d, defaulting to HLOC viz')

            viz.show(self.reconstruction, str(self.image_path))
        else:
            logger.warning(f"Selected vizualizer is not valid: {vizualizer}")


if __name__ == '__main__':

    images = Path('./data/kitti/frames/')
    output = Path('./out/test1/')
    export_name = output / 'reconstruction.ply'

    init_max_num_images = 5
    frame_skip = 1
    max_frame = 20
    slam = Pipeline()
    slam.load_data(images, output, export_name, init_max_num_images=init_max_num_images, frame_skip=frame_skip,
                   max_frame=max_frame)
    slam.run()
    slam.vizualize(vizualizer='hloc')

