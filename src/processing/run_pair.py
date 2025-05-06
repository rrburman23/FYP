from typing import Iterator, List, Tuple
import cv2
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from detectors import DetectorBase
from trackers import TrackerBase
from numbers import Real
from util.visualise import draw_detections, draw_tracks

# Configure logging for detailed debugging and error reporting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pair(
    detector: 'DetectorBase',
    tracker: 'TrackerBase',
    video_path: str | Path,
    save_output_path: str | Path | None = None,
    skip_frames: int = 1
) -> Iterator[Tuple[np.ndarray, np.ndarray, List]]:
    """
    Run a detector and tracker on a video, yielding processed frames and results.

    This function processes a video frame-by-frame, applying the specified detector to identify objects
    and the tracker to maintain object identities across frames. It validates detections, formats them
    for the tracker, and annotates frames with bounding boxes for both detections and tracks. Optionally,
    it saves the annotated video to disk using the mp4v codec for compatibility with MP4 containers.

    Args:
        detector: Object detector with a `detect(frame)` method returning `[x1, y1, x2, y2, confidence]`.
        tracker: Object tracker with an `update(frame, detections)` method returning tracks.
        video_path: Path to the input video file (MP4, MOV, or AVI).
        save_output_path: Optional path to save the annotated output video.
        skip_frames: Number of frames to skip between processing (default: 1, process every frame).

    Yields:
        Tuple containing:
        - vis_frame: Processed frame with drawn detections and tracks (NumPy array, BGR format).
        - detections: Array of validated detections `[x1, y1, x2, y2, confidence]`.
        - tracks: List of tracks, format depends on tracker (e.g., `[track_id, x1, y1, x2, y2]`).

    Raises:
        ValueError: If the video file or video writer cannot be initialized.
    """
    # Convert video_path to string for OpenCV compatibility
    video_path = str(video_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    # Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Opened video: {video_path}, FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")

    # Initialize video writer for output, if specified
    writer = None
    if save_output_path:
        save_output_path = Path(save_output_path)
        save_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use mp4v codec for compatibility with MP4 containers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(save_output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error(f"Failed to initialize video writer: {save_output_path}")
            cap.release()  # Release capture before raising error
            raise ValueError(f"Cannot initialize video writer: {save_output_path}")
        logger.info(f"Initialized video writer: {save_output_path} with mp4v codec")

    frame_count = 0
    try:
        while cap.isOpened():
            # Read frame from video
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"Frame {frame_count}: End of video reached")
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                logger.debug(f"Frame {frame_count}: Skipped due to skip_frames={skip_frames}")
                continue

            # Validate frame
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.error(f"Frame {frame_count}: Invalid frame data")
                continue

            # Run detector
            try:
                raw_detections = detector.detect(frame)
                logger.debug(f"Frame {frame_count}: Full raw detections: {raw_detections}")
                if not isinstance(raw_detections, (list, np.ndarray)):
                    logger.error(f"Frame {frame_count}: Invalid detection output type: {type(raw_detections)}")
                    continue
                # Immediate validation of raw detections
                valid_detections = []
                for det in raw_detections:
                    try:
                        if not isinstance(det, (list, np.ndarray)) or len(det) != 5:
                            logger.error(f"Frame {frame_count}: Malformed detection from detector: {det}")
                            continue
                        x1, y1, x2, y2, conf = det
                        if not (isinstance(x1, Real) and isinstance(y1, Real) and
                                isinstance(x2, Real) and isinstance(y2, Real) and
                                isinstance(conf, Real)):
                            logger.error(f"Frame {frame_count}: Invalid detection values: {det}")
                            continue
                        valid_detections.append(det)
                    except Exception as e:
                        logger.error(f"Frame {frame_count}: Unpacking detection failed: {det}, error: {e}")
                        continue
                logger.debug(f"Frame {frame_count}: Raw detections after validation: {valid_detections}")
                detections = np.array(valid_detections, dtype=np.float32) if valid_detections else np.array([], dtype=np.float32)
            except Exception as e:
                logger.error(f"Frame {frame_count}: Detector failed: {e}")
                continue

            # Validate detections and fix flipped boxes
            valid_detections = []
            for det in detections:
                try:
                    x1, y1, x2, y2 = det[:4]
                    conf = det[4] if len(det) > 4 else 1.0  # Default confidence if absent
                    logger.debug(f"Checking detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}")

                    # Auto-correct flipped bounding boxes
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    # Reject invalid detections (negative confidence)
                    if conf < 0:
                        logger.warning(f"Frame {frame_count}: Rejected detection with negative confidence: {conf}")
                        continue

                    # Clamp coordinates to frame boundaries
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # Skip zero-width or zero-height boxes
                    if x1 == x2 or y1 == y2:
                        logger.warning(f"Frame {frame_count}: Skipping zero-sized detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}")
                        continue

                    valid_detections.append([x1, y1, x2, y2, conf])
                except Exception as e:
                    logger.error(f"Frame {frame_count}: Validation failed for detection: {det}, error: {e}")
                    continue

            detections = np.array(valid_detections, dtype=np.float32)
            logger.debug(f"Frame {frame_count}: {len(valid_detections)} valid detections after validation")

            # Prepare detections for tracker
            tracker_detections = detections[:, :4].tolist() if len(detections) > 0 else []  # [x1, y1, x2, y2]

            # Run tracker
            try:
                logger.debug(f"Frame {frame_count}: Tracker input: {tracker_detections}")
                tracks = tracker.update(frame, tracker_detections)
                logger.debug(f"Frame {frame_count}: {len(tracks)} tracks, tracks={tracks}")
            except Exception as e:
                logger.error(f"Frame {frame_count}: Tracker failed: {e}")
                tracks = []

            # Format tracks for visualization and metrics
            formatted_tracks = [
                {"track_id": t[0], "bbox": t[1:5]} for t in tracks
            ] if tracks else []

            # Draw visualizations
            try:
                vis_frame = frame.copy()
                vis_frame = draw_detections(vis_frame, detections)
                vis_frame = draw_tracks(vis_frame, formatted_tracks)
            except Exception as e:
                logger.error(f"Frame {frame_count}: Visualization failed: {e}")
                vis_frame = frame.copy()

            # Write to output video
            if writer:
                try:
                    writer.write(vis_frame)
                    logger.debug(f"Frame {frame_count}: Wrote frame to output video")
                except Exception as e:
                    logger.error(f"Frame {frame_count}: Failed to write frame to video: {e}")

            logger.debug(f"Frame {frame_count}: Yielding frame with {len(detections)} detections and {len(formatted_tracks)} tracks")
            yield vis_frame, detections, formatted_tracks

    except Exception as e:
        logger.error(f"Unexpected error in run_pair: {e}")
        raise

    finally:
        # Clean up resources
        cap.release()
        if writer:
            writer.release()
            logger.info(f"Saved output video to {save_output_path}")