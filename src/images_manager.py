import os
from pathlib import Path
from src import features
import pycolmap
from src import enums
import numpy as np
from tqdm import tqdm


class ImagesManager:
    """
    This class stores the used image list and their correspondent features and matches
    Should be seen as a substitution for the database
    """

    # The path to the images on the disk
    images_path = Path("")

    # The names of the images that should be used
    frame_names = []

    # Maps an image_id to its corresponding keypoints (features) in the image
    kp_map = {}

    # Maps an image_id to the description of the corresponding keypoints in the image
    descriptor_map = {}

    # Keeps track which images have been
    managed_images = []

    # List of all image ids
    image_ids = []

    # The reconstruction object
    reconstruction = None

    # The correspondence graph
    graph = None

    # The only camera in the reconstruction (assuming SLAM)
    camera = None

    # The number of images to compare the current frame with
    # We do an exhaustive matching with the images
    # [current_img - init_max_num_images : current_img + init_max_num_images]
    init_max_num_images = 60

    def __init__(self, images_path, frame_names, reconstruction, graph, camera, init_max_num_images=60, used_extractor=enums.Extractors.ORB,
                 used_matcher=enums.Matchers.OrbHamming):
        self.images_path = images_path
        self.frame_names = frame_names
        self.used_extractor = used_extractor
        self.used_matcher = used_matcher
        self.extractor, self.matcher = features.init(used_extractor, used_matcher)
        self.kp_map = {}
        self.descriptor_map = {}
        self.image_ids = []
        self.graph = graph
        self.reconstruction = reconstruction
        self.camera = camera
        self.init_max_num_images = init_max_num_images

        # Register all images
        for image_id in tqdm(range(len(frame_names)), "Matching the corresponding images"):
            self.register_image(image_id)

        self.image_ids = sorted(self.image_ids)

        # Set the correspondences
        for image_id in range(len(frame_names)):
            self.reconstruction.images[image_id].num_observations = self.graph.num_observations_for_image(image_id)
            self.reconstruction.images[image_id].num_correspondences = self.graph.num_correspondences_for_image(image_id)


    def register_image(self, image_id):
        """
        Extract keypoints+descriptors, add image to reconstruction and graph,
        and add correspondences
        :param image_id:
        :return:
        """
        self.kp_map[image_id], self.descriptor_map[image_id] = features.detector(self.images_path,
                                                                               self.frame_names[image_id],
                                                                               self.extractor,
                                                                               self.used_extractor)
        image = pycolmap.Image(id=image_id, name=str(self.frame_names[image_id]),
                                camera_id=self.camera.camera_id)
        image.registered = False #TODO: why is this False?
        points2D = [keypoint.pt for keypoint in self.kp_map[image_id]]
        image.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
        self.reconstruction.add_image(image)
        self.graph.add_image(image_id, len(image.points2D))
        self.image_ids.append(image_id)

        for image_id2 in self.image_ids[max(0, image_id - self.init_max_num_images):image_id]:
            self.add_to_correspondence_graph(image_id, image_id2)

    def match_images(self, image_id1, image_id2):
        matches = features.matcher(self.descriptor_map[image_id1], self.descriptor_map[image_id2], self.matcher,
                                   self.used_matcher)
        # Since the first parameter for matcher is actually the query
        matches = [(match.queryIdx, match.trainIdx) for match in matches]
        return matches

    def add_to_correspondence_graph(self, image_id1, image_id2):
        matches = self.match_images(image_id1, image_id2)
        matches = np.array(matches, dtype=np.uint32)
        self.graph.add_correspondences(image_id1, image_id2, matches)

    # Check if image exists on disk
    def exists_image(self, image_id):
        # TODO should be rewritten to actual check on disk
        return 0 <= image_id < len(self.frame_names)

    # Returns a tuple of two image id´s where the first entry is the smaller one
    def ImagePairToPairId(self, image_id1, image_id2):
        if image_id1 <= image_id2:
            return (image_id1, image_id2)
        else:
            return (image_id2, image_id1)
