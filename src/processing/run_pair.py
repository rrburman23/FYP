import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Union

from detectors import YOLOv5, YOLOv3, FasterRCNN, SSDDetector
from trackers import TrackerBase
from utils.visualise import draw_detections, draw_tracks


def run_pair(
    detector: Union[YOLOv5, YOLOv3, FasterRCNN, SSDDetector],
    tracker: TrackerBase,
    video_path: str,
    show_window: bool = False,
    save_output_path: Union[str, Path] = None
) -> Generator[tuple[np.ndarray, np.ndarray, list], None, None]:
    """
    Process a video stream frame-by-frame, apply the detector and tracker,
    and yield the processed frame, raw detections, and tracking results.

    Args:
        detector (Union[YOLOv5, YOLOv3, FasterRCNN, SSDDetector]):
            Initialized object detector instance.
        tracker (TrackerBase):
            Initialized tracker instance.
        video_path (str):
            Path to input video file.
        show_window (bool, optional):
            Whether to display the processing window with OpenCV. Defaults to False.
        save_output_path (str, optional):
            If provided, saves the annotated video to this path.

    Yields:
        Generator[Tuple[frame, detections, tracks]]:
            - frame (np.ndarray): Processed video frame with annotations.
            - detections (np.ndarray): Detection boxes and confidences.
            - tracks (list): Tracker output for the frame.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"[run_pair] ❌ Video not found at: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"[run_pair] ❌ Failed to open video: {video_path}")

    print(f"[run_pair] 🔄 Processing: {video_path} Detector: {detector} Tracker: {tracker}")

    # Video writer setup
    video_writer = None
    if save_output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(str(save_output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect objects in the frame
        detections = detector.detect(frame)
        if isinstance(detections, np.ndarray):
            detections = detections.tolist()
        detections = [d[:5] for d in detections]  # Strip extra info if needed
        detections = np.array(detections, dtype=np.float32)

        # Step 2: Update tracker with detections
        tracked_objects = tracker.update(frame, detections) if detections.size > 0 else []

        # Step 3: Visualize results
        frame = draw_detections(frame, detections)
        frame = draw_tracks(frame, tracked_objects)

        # Step 4: Save output frame (if applicable)
        if video_writer:
            video_writer.write(frame)

        # Step 5: Display window if enabled
        if show_window:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        yield frame, detections, tracked_objects

    cap.release()
    if video_writer:
        video_writer.release()
    if show_window:
        cv2.destroyAllWindows()
