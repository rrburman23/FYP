"""Base class for object detectors.

All detectors must inherit from `BaseDetector` and implement the
`predict` method.

"""

from abc import ABC, abstractmethod
import cv2

class DetectorBase(ABC):
    """
    Abstract base class for all detection models. 
    All detection algorithms should inherit from this class and implement the detect method.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialises the detection model.
        This method should be implemented in each subclass to load the model.
        """
        pass

    @abstractmethod
    def detect(self, frame: cv2.Mat):
        """
        Detect objects in the given frame.

        :param frame: The input frame (image) for detection.
        :return: List of bounding boxes for detected objects.
        """
        pass

