import cv2
from trackers.base import TrackerBase

class OpenCVTracker(TrackerBase):
    def __init__(self):
        self.tracker = cv2.TrackerKCF_create()

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        return success, bbox

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)
