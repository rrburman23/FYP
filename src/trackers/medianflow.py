"""
OpenCV MedianFlow Tracker.
Requires:
    pip install opencv-contrib-python
"""

import cv2
from trackers.base import TrackerBase


def create_medianflow_tracker():
    """
    Creates a MedianFlow tracker compatible with different OpenCV versions.
    """
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerMedianFlow_create()
    elif hasattr(cv2, 'TrackerMedianFlow_create'):
        return cv2.TrackerMedianFlow_create()
    else:
        raise AttributeError("MedianFlow tracker is not available in your OpenCV installation.")


class MedianFlowTracker(TrackerBase):
    """
    Wrapper for OpenCV's MedianFlow tracker.
    """

    def __init__(self):
        self.tracker = create_medianflow_tracker()
        self.initialized = False
        self.track_id = 0

    def init(self, frame, bbox):
        """
        Initialize the tracker with a frame and bounding box.
        """
        self.tracker = create_medianflow_tracker()
        self.initialized = self.tracker.init(frame, tuple(bbox))

    def update(self, frame, detections):
        """
        Update tracker state. If uninitialized, it starts with first detection.

        Args:
            frame (np.ndarray): Frame in BGR format.
            detections (List): [[x1, y1, x2, y2, score], ...]

        Returns:
            List: [[track_id, x1, y1, x2, y2], ...]
        """
        if not self.initialized:
            if len(detections) == 0:
                return []
            # Use first detection to initialize
            x1, y1, x2, y2, _ = detections[0]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            self.init(frame, bbox)
            return [[self.track_id, int(x1), int(y1), int(x2), int(y2)]]

        success, bbox = self.tracker.update(frame)
        if not success:
            return []

        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        return [[self.track_id, x1, y1, x2, y2]]
