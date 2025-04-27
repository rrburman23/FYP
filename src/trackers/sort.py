# sort.py - SORT Algorithm using Kalman Filter for Tracking
# This code integrates the Kalman filter directly into the SORT tracker logic.

from filterpy.kalman import KalmanFilter
import numpy as np
from trackers.base import TrackerBase

class KalmanTracker:
    """ Kalman Filter based tracker for object tracking. """

    def __init__(self, iou_threshold: float = 0.1, max_age: int = 30):
        """
        Initializes the Kalman Tracker with tracking parameters.

        :param iou_threshold: Threshold for Intersection over Union (IoU) to consider a detection matched.
        :param max_age: The maximum number of frames to keep an unmatched track.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []

    def initialize_kalman(self):
        """ Initializes the Kalman filter for a new track. """
        kf = KalmanFilter(dim_x=4, dim_z=4)
        # State transition matrix (constant velocity model)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        # Measurement matrix
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        # Initial uncertainty (high initial uncertainty)
        kf.P *= 1000.
        # Process noise
        kf.Q = np.eye(4)
        # Measurement noise
        kf.R = np.eye(4)
        return kf

    def update(self, detections: np.ndarray) -> list:
        """
        Update the tracker with new detections.

        :param detections: Array of bounding boxes in the format (x1, y1, x2, y2).
        :return: Updated list of tracked objects.
        """
        det_boxes = detections  # Detections should be a numpy array (Nx4)

        # Predict the next state for all existing tracks
        for trk in self.tracks:
            trk.predict()

        unmatched_trks = set(range(len(self.tracks)))
        # Associate detections with existing tracks (using IOU matching)
        for det in det_boxes:
            if not unmatched_trks:
                # If no unmatched tracks, add a new track
                self.tracks.append(_Track(det))
                continue

            # Compute IoU with each unmatched track
            ious = [self.box_iou(det, self.tracks[i].bbox) for i in unmatched_trks]
            if not ious:
                # Edge-case: No tracks to compare - spawn new track
                self.tracks.append(_Track(det))
                continue

            best_idx = int(np.argmax(ious))  # Select the track with the best match
            best_trk_id = list(unmatched_trks)[best_idx]
            best_iou = ious[best_idx]

            if best_iou >= self.iou_threshold:
                # If the match is good enough, update the track
                self.tracks[best_trk_id].update(det)
                unmatched_trks.remove(best_trk_id)
            else:
                # If no match, spawn a new track
                self.tracks.append(_Track(det))

        # Output the tracked objects as a list of (id, x, y, w, h)
        outputs = [(t.id, *t.current_bbox()) for t in self.tracks]

        # Remove tracks that have aged out (no detection for max_age frames)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return outputs

    def box_iou(self, box1, box2):
        """ Calculate the Intersection over Union (IoU) of two bounding boxes. """
        x1, y1, x2, y2 = box1
        xx1, yy1, xx2, yy2 = box2

        inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (xx2 - xx1) * (yy2 - yy1)

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou


class _Track:
    """ Internal Track class for managing individual tracks. """

    def __init__(self, bbox):
        self.id = np.random.randint(0, 10000)  # Random ID for the track
        self.bbox = bbox  # Bounding box
        self.time_since_update = 0  # Time since last update
        self.kf = KalmanTracker().initialize_kalman()  # Initialize Kalman filter
        self.kf.x = np.array([bbox[0], bbox[1], 0, 0])  # Initialize with detection position

    def update(self, bbox):
        """ Update the track with a new detection. """
        self.kf.update(np.array([bbox[0], bbox[1], 0, 0]))  # Update Kalman filter with detection
        self.bbox = self.kf.x[:4]  # Update track bounding box

    def predict(self):
        """ Predict the next state of the track using Kalman filter. """
        self.kf.predict()
        self.time_since_update += 1

    def current_bbox(self):
        """ Return the current predicted bounding box. """
        return self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]


class SORTTracker(TrackerBase):
    """ SORT Tracker class integrating Kalman filter tracking logic. """

    def __init__(self):
        """
        Initialize the SORT tracker.
        """
        self.kalman_tracker = KalmanTracker()

    def update(self, frame, boxes):
        """
        Update the tracker with new detections.

        :param frame: The current frame.
        :param boxes: List of bounding boxes.
        :return: Updated list of tracked boxes.
        """
        boxes = np.array(boxes)  # Ensure detections are in the correct numpy format
        return self.kalman_tracker.update(boxes)

    def init(self, frame, bbox):
        """
        The Kalman tracker doesn't require manual initialization.
        """
        pass
