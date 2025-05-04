import cv2
import numpy as np
from typing import Union
from detectors import YOLOv5, YOLOv3, FasterRCNN, SSDDetector
from trackers import TrackerBase
from utils.visualise import draw_detections, draw_tracks 
from pathlib import Path

def run_pair(
    detector: Union[YOLOv5, YOLOv3, FasterRCNN, SSDDetector],
    tracker: TrackerBase,
    video_path: str,
    show_window: bool = False
) -> None:
    """
    Process a video frame-by-frame, apply the detector and tracker, and optionally display the results.
    """
    
    video_path = Path(video_path)  # Convert video_path to a Path object for proper file handling

    # Check if the video file exists before attempting to open it
    if not video_path.exists():
        print(f"[run_pair] ❌ Video file not found at: {video_path}")
        raise FileNotFoundError(f"[run_pair] ❌ Video file not found at: {video_path}")
    else:
        print(f"[run_pair] ✅ Video file exists at: {video_path}")
        cap = cv2.VideoCapture(str(video_path))  # Convert Path to string and open video

    if not cap.isOpened():
        raise FileNotFoundError(f"[run_pair] ❌ Failed to open video: {video_path}")

    print(f"[run_pair] 🔄 Processing video: {video_path} Detector: {detector} Tracker: {tracker}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[run_pair] ✅ Finished processing video.")
            break

        # Run object detection
        detections = detector.detect(frame)  # Expected format: List[Tuple[x1, y1, x2, y2, conf]]

        # Ensure detections are in the correct format (List[Tuple[x1, y1, x2, y2, conf]])
        if isinstance(detections, np.ndarray):
            detections = detections.tolist()

        # DEBUG: Check the format of the detections
        # print("[run_pair] Detections format:", detections)

        # Ensure each detection has 5 elements (x1, y1, x2, y2, confidence)
        detections = [detection[:5] for detection in detections]  # Strip extra data if necessary
        
        # Ensure detections are correctly formatted for DeepSort (as numpy array)
        detections = np.array(detections, dtype=np.float32)

        # Check for empty or invalid detections
        if detections.size == 0:
            tracked_objects = []  # No detections, empty track list
        else:
            tracked_objects = tracker.update(frame, detections)  # Update tracker with valid detections

        # Draw tracked objects and detections on the frame
        frame = draw_detections(frame, detections)  # Annotate frame with detections
        frame = draw_tracks(frame, tracked_objects)  # Annotate frame with tracks

        if show_window:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Yield frame along with detections and tracks
        yield frame, detections, tracked_objects

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
