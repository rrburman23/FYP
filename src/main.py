"""
main.py
-------
Command‑line entry point for batch processing a single video with a
detector + tracker pair (no GUI).

Usage example:
    python -m src.main --det YOLOv5 --trk SORT --video data/test.mp4
"""

import argparse
import cv2
from utils.paths import load_config
from processing.run_pair import run_pair

# dynamic maps for quick look‑up
DETECTORS = {
    "SSD":       "detectors.ssd.SSD",
    "YOLOv5":    "detectors.yolov5.YOLOv5",
    "FasterRCNN":"detectors.fasterrcnn.FasterRCNN",
    "YOLOv3":    "detectors.yolo.YOLO",
    "RetinaNet": "detectors.retinanet.RetinaNet",
}
TRACKERS = {
    "SORT":     "trackers.sort.SORTTracker",
    "DeepSORT": "trackers.deepsort.DeepSORTTracker",
    "ByteTrack":"trackers.bytetrack.ByteTrackTracker",
    "OpenCV":   "trackers.opencv.OpenCVTracker",
}

def import_from(path: str):
    """Import class from 'module.ClassName' string."""
    module_name, cls_name = path.rsplit(".", 1)
    mod = __import__(module_name, fromlist=[cls_name])
    return getattr(mod, cls_name)

def main():
    parser = argparse.ArgumentParser(description="UAV detector & tracker batch run")
    parser.add_argument("--det",  choices=DETECTORS.keys(), default="SSD")
    parser.add_argument("--trk",  choices=TRACKERS.keys(), default="Kalman")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--cfg",   type=str, help="Optional YAML config override")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    Detector = import_from(DETECTORS[args.det])
    Tracker  = import_from(TRACKERS[args.trk])

    detector = Detector(conf_thresh=cfg["processing"]["confidence_threshold"]) \
               if args.det == "SSD" else Detector()
    tracker  = Tracker()

    for _, _ in run_pair(detector, tracker, args.video, show_window=True):
        pass      # run_pair shows frames itself when show_window=True

if __name__ == "__main__":
    main()
