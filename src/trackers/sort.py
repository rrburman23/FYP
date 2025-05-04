"""
Kalman Filter-based SORT tracker for simple multi-object tracking.
"""

from filterpy.kalman import KalmanFilter
import numpy as np
from trackers.base import TrackerBase


class _Track:
    """Internal class to manage single track instance."""

    def __init__(self, bbox):
        self.id = np.random.randint(10000)
        self.kf = self.initialize_kalman()
        self.kf.x[:2, 0] = bbox[:2]  # Initialize position
        self.bbox = bbox
        self.time_since_update = 0

    def initialize_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000.
        kf.Q = np.eye(4)
        kf.R = np.eye(2)
        return kf

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox):
        self.kf.update(np.array(bbox[:2]))  # Update Kalman filter with new detection
        self.bbox[:2] = self.kf.x[:2].flatten()  # Correctly assign the 2x1 Kalman output to a 1D array
        self.time_since_update = 0


    def current_bbox(self):
        x, y = self.kf.x[:2]
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        return int(x), int(y), int(x + w), int(y + h)


class KalmanTracker:
    """Manages all SORT tracks."""

    def __init__(self, iou_threshold=0.1, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []

    def update(self, detections):
        updated_tracks = []

        for track in self.tracks:
            track.predict()

        for det in detections:
            matched = False
            for track in self.tracks:
                iou = self.iou(det, track.bbox)
                if iou > self.iou_threshold:
                    track.update(det)
                    matched = True
                    break
            if not matched:
                self.tracks.append(_Track(det))

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        for t in self.tracks:
            x1, y1, x2, y2 = t.current_bbox()
            updated_tracks.append([t.id, x1, y1, x2, y2])

        return updated_tracks

    def iou(self, boxA, boxB):
        x1, y1, x2, y2 = boxA[:4]
        xx1, yy1, xx2, yy2 = boxB[:4]

        inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
        boxA_area = (x2 - x1) * (y2 - y1)
        boxB_area = (xx2 - xx1) * (yy2 - yy1)
        return inter_area / float(boxA_area + boxB_area - inter_area)


class SORTTracker(TrackerBase):
    """
    SORT Tracker using internal Kalman filtering.
    """

    def __init__(self):
        self.tracker = KalmanTracker()

    def init(self, frame, bbox):
        # SORT doesn't require manual init
        pass

    def update(self, frame, boxes):
        """
        Args:
            frame: np.ndarray (unused but required by base)
            boxes: List of [x1, y1, x2, y2]

        Returns:
            List of [track_id, x1, y1, x2, y2]
        """
        return self.tracker.update(boxes)
