import cv2
from utils.paths import load_config

# Define the run_pair function directly here
def run_pair(detector, tracker, video_path, show_window=False):
    """
    Process a video frame by frame, apply the detector and tracker, and stream the annotated frames.

    Args:
    - detector: Object of the chosen detector (e.g., YOLOv5, SSD)
    - tracker: Object of the chosen tracker (e.g., SORT, Kalman)
    - video_path: Path to the input video
    - show_window: Boolean flag to show the processed frames in a window
    """
    
    # Open video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the detector to the frame
        detections = detector.detect(frame)

        # Update tracker with detected bounding boxes
        tracker.update(detections)

        # Annotate the frame with tracking information (e.g., bounding boxes)
        for track in tracker.get_tracks():
            x1, y1, x2, y2 = track  # Assuming tracker returns bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Show the frame with annotations if requested
        if show_window:
            cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()