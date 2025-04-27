"""
deepsort.py
===========
Deep SORT wrapper that conforms to TrackerBase.

Requires: pip install deep-sort-realtime
"""

from __future__ import annotations
from typing import List
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from trackers.base import TrackerBase      # adjust if your base file is named differently


class DeepSORTTracker(TrackerBase):
    """Appearance‑aware multi‑object tracker (Deep SORT)."""

    def __init__(self,
                 max_age: int = 30,
                 iou_threshold: float = 0.3) -> None:
        # n_init=1 means a track is confirmed after a single hit
        self.ds = DeepSort(max_age=max_age,
                           n_init=1,
                           nms_max_overlap=iou_threshold)

    # Deep SORT auto‑initialises; we keep an empty stub for API consistency
    def init(self, frame, bbox):
        pass

    def update(self,
               frame,
               detections: List[list]) -> List[list]:
        """
        Args
        ----
        frame       : current BGR frame (array) - Deep-SORT can use it for re-ID.
        detections  : [[x1,y1,x2,y2,conf], …]  - pixel coordinates.

        Returns
        -------
        [[track_id, x1, y1, x2, y2], …] for all confirmed tracks.
        """
        dets = np.asarray([d[:5] for d in detections], dtype=float) if detections else np.empty((0, 5))
        tracks = self.ds.update_tracks(dets, frame=frame)

        output = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            tid = trk.track_id
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            output.append([tid, x1, y1, x2, y2])
        return output
