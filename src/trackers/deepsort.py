"""
DeepSORT wrapper tracker.

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

    Tracks are re-identified using visual similarity and motion.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3) -> None:
        self.ds = DeepSort(max_age=max_age, n_init=1, nms_max_overlap=iou_threshold)

    def init(self, frame, bbox):
        # DeepSORT auto-initializes internally.
        pass

    def update(self, frame, detections: List[list]) -> List[list]:
        """
        Update tracking with detections.

        Args:
            frame (np.ndarray): Current BGR frame.
            detections (List): [[x1, y1, x2, y2, score], ...]

        Returns:
            List: [[track_id, x1, y1, x2, y2], ...]
        """
        # DeepSORT expects: [([x1, y1, x2, y2], score, class_id), ...]
        if len(detections) > 0:
            formatted_detections = [
                ([float(d[0]), float(d[1]), float(d[2]), float(d[3])], float(d[4]), 0)
                for d in detections
            ]
        else:
            formatted_detections = []

        tracks = self.ds.update_tracks(formatted_detections, frame=frame)

        output = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            output.append([trk.track_id, x1, y1, x2, y2])
        return output
