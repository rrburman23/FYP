import cv2
from trackers.base import TrackerBase

class MOSSETracker(TrackerBase):
    def __init__(self):
        # Check if MOSSE tracker is available in the current OpenCV installation
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            self.tracker = cv2.legacy.TrackerMOSSE_create()  # Use legacy version if available
        elif hasattr(cv2, 'TrackerMOSSE_create'):
            self.tracker = cv2.TrackerMOSSE_create()  # Use non-legacy version
        else:
            raise AttributeError("MOSSE tracker is not available in your OpenCV installation.")
        self.initialized = False
        self.track_id = 0

    def init(self, frame, bbox):
        self.initialized = self.tracker.init(frame, tuple(bbox))

    def update(self, frame, detections):
        if not self.initialized:
            if len(detections) == 0:
                return []
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
