from bytetrack import BYTETracker
from trackers.base import TrackerBase

class ByteTrackTracker(TrackerBase):
    def __init__(self):
        self.bytetracker = BYTETracker()

    def update(self, frame, boxes):
        return self.bytetracker.update(boxes)
