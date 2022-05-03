import os
from pathlib import Path
from src import features
import pycolmap
import enums


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

    def __init__(self, images_path, frame_names, used_extractor=enums.Extractors.ORB,
                 used_matcher=enums.Matchers.OrbHamming):
        self.images_path = images_path
        self.frame_names = frame_names
        self.used_extractor = used_extractor
        self.used_matcher = used_matcher
        self.extractor, self.matcher = features.init(used_extractor, used_matcher)
        self.kp_map = {}
        self.detector_map = {}

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

    def match_images(self, image_id1, image_id2):
        matches = features.matcher(self.detector_map[image_id2], self.detector_map[image_id1], self.matcher,
                                   self.used_matcher)
        return matches

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
