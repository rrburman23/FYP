"""
main.py
-------
Command‑line entry point for batch processing a single video with a
detector + tracker pair (no GUI).

Usage example:
    python -m src.main --det YOLOv5 --trk MedianFlow --video data/testFootage.mp4
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
    " ": "detectors. . ",
}
TRACKERS = {
    "SORT":     "trackers.sort.SORTTracker",
    "DeepSORT": "trackers.deepsort.DeepSORTTracker",
    "OpenCV":   "trackers.opencv.OpenCVTracker",
    "MedianFlow":"trackers.medianflow.MedianFlowTracker",
    "GOTURN":   "trackers.goturn.GOTURNTracker",
}

def import_from(path: str):
    """Import class from 'module.Class' notation"""
    parts = path.split(".")
    mod = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
    return getattr(mod, parts[-1])

def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Run detector-tracker pair on video")
    parser.add_argument("--det", required=True, choices=DETECTORS.keys(), help="Detector name")
    parser.add_argument("--trk", required=True, choices=TRACKERS.keys(), help="Tracker name")
    parser.add_argument("--video", required=True, type=str, help="Path to video file")

    args = parser.parse_args()

    # Config and paths
    cfg = load_config()
    video_path = args.video

    # Detector factory
    Detector = import_from(DETECTORS[args.det])
    detector = Detector(conf_thresh=cfg["processing"]["confidence_threshold"])

    # Tracker factory
    Tracker = import_from(TRACKERS[args.trk])
    tracker = Tracker()

    # Process video
    for _, frm in run_pair(detector, tracker, video_path):
        cv2.imshow("frame", frm)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
