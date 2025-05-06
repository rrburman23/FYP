import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from .base import TrackerBase

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = np.array(bbox).reshape(4, 1)

class SORTTracker(TrackerBase):
    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.1):
        """
        Initialize SORT tracker.

        :param max_age: Maximum frames before track termination
        :param min_hits: Minimum detections to start a track
        :param iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def init(self, frame: np.ndarray, bbox: list):
        """
        Placeholder for abstract method from TrackerBase.
        SORT does not require initialization with a specific bounding box.
        """
        pass


    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Update tracker with new detections.

        :param frame: Input frame (BGR)
        :param detections: List of [x1, y1, x2, y2]
        :return: List of tracks [track_id, x1, y1, x2, y2]
        """
        # Predict existing tracks
        for track in self.tracks:
            track.kf.predict()
            track.bbox = track.kf.x[:4].flatten()
            track.age += 1
            track.time_since_update += 1

        # Convert detections to numpy for processing
        detections = np.array(detections, dtype=np.float32) if detections else np.empty((0, 4))

        # Compute IOU between tracks and detections
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = self._iou(track.bbox, det)

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            # Update matched tracks
            unmatched_detections = set(range(len(detections)))
            unmatched_tracks = set(range(len(self.tracks)))

            for t, d in zip(row_ind, col_ind):
                if iou_matrix[t, d] > self.iou_threshold:
                    self.tracks[t].kf.update(detections[d])
                    self.tracks[t].bbox = detections[d]
                    self.tracks[t].hits += 1
                    self.tracks[t].time_since_update = 0
                    unmatched_detections.discard(d)
                    unmatched_tracks.discard(t)

            # Mark unmatched tracks
            for t in unmatched_tracks:
                self.tracks[t].time_since_update += 1

            # Create new tracks for unmatched detections
            for d in unmatched_detections:
                new_track = Track(detections[d], self.next_id)
                self.next_id += 1
                self.tracks.append(new_track)
        else:
            # Create new tracks for all detections
            for det in detections:
                new_track = Track(det, self.next_id)
                self.next_id += 1
                self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return tracks that meet minimum hits
        output = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.time_since_update == 0:
                output.append([track.track_id] + track.bbox.tolist())

        return output

    def _iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IOU between two bounding boxes.

        :param bbox1: [x1, y1, x2, y2]
        :param bbox2: [x1, y1, x2, y2]
        :return: IOU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0