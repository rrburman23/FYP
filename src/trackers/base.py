"""Abstract tracker interface."""
from abc import ABC, abstractmethod
import cv2

class TrackerBase(ABC):
    """
    Abstract base class for all tracking models.
    All tracking algorithms should inherit from this class and implement the update and init methods.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialises the tracking model.
        This method should be implemented in each subclass to set up the tracker.
        """
        pass

    @abstractmethod
    def update(self, frame: cv2.Mat, boxes: list):
        """
        Update the tracker with new frame and bounding boxes.

        :param frame: The current frame (image) for tracking.
        :param boxes: List of bounding boxes of objects to track.
        :return: Updated list of tracked objects.
        """
        pass

    @abstractmethod
    def init(self, frame: cv2.Mat, bbox: list):
        """
        Initialise the tracker with the initial frame and bounding box.

        :param frame: The first frame (image) for tracking.
        :param bbox: The bounding box coordinates (x, y, w, h).
        """
        pass
