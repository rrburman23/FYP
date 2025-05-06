"""
DeepSORT wrapper tracker.

This module provides a wrapper for the DeepSORT tracker, implementing the TrackerBase
interface. DeepSORT combines motion prediction with appearance-based re-identification
for robust multi-object tracking.

Requires:
    pip install deep-sort-realtime
"""

from __future__ import annotations
from typing import List
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from trackers.base import TrackerBase

class DeepSORTTracker(TrackerBase):
    """
    DeepSORT tracker with appearance feature embedding.

    Tracks are re-identified using visual similarity and motion, providing robust
    multi-object tracking across occlusions and camera changes.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3) -> None:
        """
        Initialize the DeepSORT tracker.

        Args:
            max_age: Maximum frames a track can persist without updates.
            iou_threshold: Minimum IOU for matching detections to tracks.
        """
        self.ds = DeepSort(max_age=max_age, n_init=1, nms_max_overlap=iou_threshold)

    def init(self, frame: np.ndarray, bbox: List[float]) -> None:
        """
        Initialize the tracker with a frame and bounding box (not used in DeepSORT).

        Args:
            frame: Initial video frame (BGR format).
            bbox: Initial bounding box [x, y, w, h].
        """
        pass  # DeepSORT auto-initializes internally

    def update(self, frame: np.ndarray, detections: List[List[float]]) -> List[dict]:
        """
        Update tracking with new detections.

        Args:
            frame: Current video frame (BGR format).
            detections: List of detections [[x1, y1, x2, y2, score], ...].

        Returns:
            List of dictionaries with keys 'track_id' and 'bbox' [x1, y1, x2, y2].
        """
        # DeepSORT expects: [([x1, y1, x2, y2], score, class_id), ...]
        formatted_detections = [
            ([float(d[0]), float(d[1]), float(d[2]), float(d[3])], float(d[4]), 0)
            for d in detections
        ] if detections else []

        tracks = self.ds.update_tracks(formatted_detections, frame=frame)

        return [
            {"track_id": trk.track_id, "bbox": list(map(int, trk.to_ltrb()))}
            for trk in tracks if trk.is_confirmed()
        ]