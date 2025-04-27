from __future__ import annotations
from typing import List, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection‑over‑Union for two bboxes (x1,y1,x2,y2)."""
    xx1, yy1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    xx2, yy2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    w, h = np.maximum(0., xx2 - xx1), np.maximum(0., yy2 - yy1)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Internal track object
# ──────────────────────────────────────────────────────────────────────────────
class _Track:
    """A single Kalman‑filter track."""
    _next_id = 1

    def __init__(self, bbox: np.ndarray):
        # Assign a unique ID
        self.id = _Track._next_id
        _Track._next_id += 1

        # Constant‑velocity Kalman filter: 7‑D state, 4‑D measurement
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self._configure_matrices()

        # Initialise state from first detection
        self._init_state(bbox)

        self.time_since_update = 0  # frames since last matched detection
        self.bbox = bbox.copy()    # last observation

    # --------------------------------------------------------------------- #
    def _configure_matrices(self) -> None:
        """Set transition (F) and measurement (H) matrices."""
        self.kf.F = np.eye(7)
        # Top‑left 3×3 identity links position/scale to their velocities.
        # Rows 0‑2 (x, y, s) depend on cols 4‑6 (vx, vy, vs).
        self.kf.F[:3, 4:] = np.eye(3)

        # Measurement observes x, y, s, r directly.
        self.kf.H[:4, :4] = np.eye(4)

        # Process / measurement noise (tuned coarsely)
        self.kf.R *= 10.
        self.kf.P *= 50.

    # --------------------------------------------------------------------- #
    def _init_state(self, bbox: np.ndarray) -> None:
        """Convert bbox to (x,y,s,r) and place into filter state."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        self.kf.x[:4] = np.array([x1, y1, s, r]).reshape((4, 1))

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def predict(self) -> None:
        """Advance state by one frame."""
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox: np.ndarray) -> None:
        """Correct state with an observed bbox."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        z = np.array([x1, y1, w * h, w / (h + 1e-6)])
        self.kf.update(z)
        self.time_since_update = 0
        self.bbox = bbox.copy()

    def current_bbox(self) -> Tuple[int, int, int, int]:
        """Return the current predicted bbox as integers."""
        x, y, s, r = self.kf.x[:4].reshape(-1)

        # Clamp to avoid negative / zero values that lead to NaNs
        s = max(float(s), 1e-3)
        r = max(float(r), 1e-3)

        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        return int(x), int(y), int(x + w), int(y + h)


# ──────────────────────────────────────────────────────────────────────────────
# Main tracker class
# ──────────────────────────────────────────────────────────────────────────────
class KalmanTracker:
    """
    Greedy IoU multi‑object tracker.

    Args
    ----
    iou_threshold : float
        Minimum IoU to match a detection to an existing track.
    max_age : int
        Frames to keep a track alive without new detections.
    """

    def __init__(self, iou_threshold: float = 0.1, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[_Track] = []

    # --------------------------------------------------------------------- #
    def update(self,
               detections: List[List]) -> List[Tuple[int, int, int, int, int]]:
        """
        Update with detections for one frame.

        Args
        ----
        detections : list
            Each item is [x1, y1, x2, y2, conf].

        Returns
        -------
        list
            (track_id, x1, y1, x2, y2) for all active tracks.
        """
        det_boxes = np.array([d[:4] for d in detections]) if detections else np.empty((0, 4))

        # 1. Predict new state for every existing track.
        for trk in self.tracks:
            trk.predict()

        # 2. Greedy matching of detections to tracks.
        unmatched_trks = set(range(len(self.tracks)))
        for det in det_boxes:
            if not unmatched_trks:
                # No tracks left to match → start a new one.
                self.tracks.append(_Track(det))
                continue

            # Compute IoU with each still‑unmatched track.
            ious = [box_iou(det, self.tracks[i].bbox) for i in unmatched_trks]
            if not ious:
                # Edge‑case: no tracks to compare - start new track
                self.tracks.append(_Track(det))
                continue

            best_idx = int(np.argmax(ious))  # Fixed: define best_idx after ious calculation

            best_trk_id = list(unmatched_trks)[best_idx]
            best_iou = ious[best_idx]

            if best_iou >= self.iou_threshold:
                # Found a suitable match.
                self.tracks[best_trk_id].update(det)
                unmatched_trks.remove(best_trk_id)
            else:
                # Not similar enough → spawn new track.
                self.tracks.append(_Track(det))

        # 3. Build outputs BEFORE pruning, so even one‑frame tracks are returned.
        outputs = [(t.id, *t.current_bbox()) for t in self.tracks]

        # 4. Drop tracks that have aged out.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return outputs
