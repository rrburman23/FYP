"""
Abstract tracker interface.

Defines the base class that all tracking implementations should inherit from.
"""

from abc import ABC, abstractmethod
import cv2


class TrackerBase(ABC):
    """
    Abstract base class for all tracking models.
    All custom trackers must implement this interface.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the tracker.
        Subclasses should configure any internal tracking models or parameters.
        """
        pass

    @abstractmethod
    def update(self, frame: cv2.Mat, boxes: list) -> list:
        """
        Update the tracker with a new frame and object detections.

        Args:
            frame (cv2.Mat): Current video frame.
            boxes (list): Detected bounding boxes [[x1, y1, x2, y2, score], ...].

        Returns:
            list: Tracked object bounding boxes with IDs.
        """
        pass

    @abstractmethod
    def init(self, frame: cv2.Mat, bbox: list):
        """
        Optional initialisation of tracker with first frame and bounding box.

        Args:
            frame (cv2.Mat): First video frame.
            bbox (list): Initial bounding box [x, y, w, h].
        """
        pass
