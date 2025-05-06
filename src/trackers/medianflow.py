"""
MedianFlow tracker implementation.

This module provides a wrapper for OpenCV's MedianFlow tracker, implementing the TrackerBase
interface for integration with the detection-tracking pipeline. It supports single-object
tracking and initializes with the first detection provided.
"""

import cv2
import numpy as np
from trackers.base import TrackerBase
from typing import List, Dict

def create_medianflow_tracker() -> 'cv2.Tracker':
    """
    Create a MedianFlow tracker compatible with different OpenCV versions.

    Returns:
        OpenCV MedianFlow tracker instance.
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

    Tracks a single object using the MedianFlow algorithm, which is robust to small
    motion changes but limited to one object.
    """

    def __init__(self):
        """
        Initialize the MedianFlow tracker.
        """
        self.tracker = create_medianflow_tracker()
        self.initialized = False
        self.track_id = 0

    def init(self, frame: np.ndarray, bbox: List[float]) -> None:
        """
        Initialize the tracker with a frame and bounding box.

        Args:
            frame: Initial video frame (BGR format).
            bbox: Initial bounding box [x, y, w, h].
        """
        self.initialized = self.tracker.init(frame, tuple(bbox))

    def update(self, frame: np.ndarray, detections: List[List[float]]) -> List[Dict]:
        """
        Update the tracker state with new frame and detections.

        If the tracker is not initialized, it uses the first detection to start tracking.
        Returns the current track position in dictionary format for metrics compatibility.

        Args:
            frame: Current video frame (BGR format).
            detections: List of detections [[x1, y1, x2, y2, score], ...].

        Returns:
            List of dictionaries with keys 'track_id' and 'bbox' [x1, y1, x2, y2].
        """
        if not self.initialized:
            if len(detections) == 0:
                return []
            x1, y1, x2, y2, _ = detections[0]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            self.init(frame, bbox)
            return [{"track_id": self.track_id, "bbox": [int(x1), int(y1), int(x2), int(y2)]}]

        success, bbox = self.tracker.update(frame)
        if not success:
            return []

        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        return [{"track_id": self.track_id, "bbox": [x1, y1, x2, y2]}]