import os
from pathlib import Path
from src import features
import pycolmap
from src import enums
import numpy as np


class ImagesManager:
    """
    This class stores the used image list and their correspondent features and matches
    Should be seen as a substitiution for the database
    """

    # The path to the images on the disk
    images_path = Path("")

    # The names of the images that should be used
    frame_names = []

    # Maps an image_id to its corresponding keypoints (features) in the image
    kp_map = {}

    # Maps an image_id to the description of the corresponding keypoints in the image
    detector_map = {}

    # Keeps track which images have been
    managed_images = []

    # The recontruction object
    reconstruction = None

    # The correspondence graph
    graph = None

    # The only camera in the reconstruction (assuming SLAM)
    camera = None

    def __init__(self, images_path, frame_names, reconstruction, graph, camera, used_extractor=enums.Extractors.ORB,
                 used_matcher=enums.Matchers.OrbHamming):
        self.images_path = images_path
        self.frame_names = frame_names
        self.used_extractor = used_extractor
        self.used_matcher = used_matcher
        self.extractor, self.matcher = features.init(used_extractor, used_matcher)
        self.kp_map = {}
        self.detector_map = {}
        self.graph = graph
        self.reconstruction = reconstruction
        self.camera = camera

        # Register all images
        for image_id in range(len(frame_names)):
            self.register_image(image_id)

    def register_image(self, image_id):
        """
        :param image_id:
        :return:
        """
        self.kp_map[image_id], self.detector_map[image_id] = features.detector(self.images_path,
                                                                               self.frame_names[image_id],
                                                                               self.extractor,
                                                                               self.used_extractor)
        image = pycolmap.Image(id=image_id, name=str(self.frame_names[image_id]),
                                camera_id=self.camera.camera_id)
        image.registered = False
        points2D = [keypoint.pt for keypoint in self.kp_map[image_id]]
        image.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
        self.reconstruction.add_image(image)
        self.graph.add_image(image_id, len(image.points2D))

    def match_images(self, image_id1, image_id2):
        matches = features.matcher(self.detector_map[image_id1], self.detector_map[image_id2], self.matcher,
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
