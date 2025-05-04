import cv2
from typing import List, Tuple

# Predefined cycle of BGR colors for drawing bounding boxes and track IDs
COLOURS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255)  # Yellow
]

def adjust_bbox(x1, y1, x2, y2, shrink_factor=0.1) -> Tuple[int, int, int, int]:
    """
    Shrink the bounding box by a certain factor.
    Args:
        x1, y1 (int): Top-left corner of the bounding box.
        x2, y2 (int): Bottom-right corner of the bounding box.
        shrink_factor (float): The percentage by which to shrink the box (default 5%).
    Returns:
        Tuple[int, int, int, int]: Adjusted coordinates of the bounding box.
    """
    width, height = x2 - x1, y2 - y1
    shrink_width = int(width * shrink_factor)
    shrink_height = int(height * shrink_factor)

    # Shrink the box equally from all sides
    x1_adjusted = x1 + shrink_width
    y1_adjusted = y1 + shrink_height
    x2_adjusted = x2 - shrink_width
    y2_adjusted = y2 - shrink_height

    # Ensure the box doesn't go out of bounds
    x1_adjusted = max(0, x1_adjusted)
    y1_adjusted = max(0, y1_adjusted)
    x2_adjusted = min(x2, x2_adjusted)
    y2_adjusted = min(y2, y2_adjusted)

    return x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted

def draw_detections(frame, detections: List[Tuple[int, int, int, int, float]]) -> 'np.ndarray':
    """
    Draw bounding boxes for object detections on the frame. This shows the detection result before tracking.

    Args:
        frame (np.ndarray): The current BGR image frame.
        detections (List[Tuple[int, int, int, int, float]]): List of detections in format:
            (x1, y1, x2, y2, confidence), where (x1, y1) are the top-left coordinates of the bounding box,
            (x2, y2) are the bottom-right coordinates, and confidence is the detection confidence score.

    Returns:
        np.ndarray: The frame with detection bounding boxes drawn.
    """
    for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
        # Adjust bounding box to be tighter
        x1, y1, x2, y2 = adjust_bbox(x1, y1, x2, y2)

        # Cycle through predefined colors for each detection
        colour = COLOURS[idx % len(COLOURS)]
        
        # Draw the bounding box for detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 1)
        
        # Display confidence score near the top-left corner of the box
        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    
    return frame

def draw_tracks(frame, tracks: List[Tuple[int, int, int, int, int]]) -> 'np.ndarray':
    """
    Draw tracking bounding boxes and track IDs on the frame. This shows the tracked objects after detection.

    Args:
        frame (np.ndarray): The current BGR image frame.
        tracks (List[Tuple[int, int, int, int, int]]): List of tracks in format:
            (track_id, x1, y1, x2, y2), where (x1, y1) are the top-left coordinates of the tracking box,
            (x2, y2) are the bottom-right coordinates, and track_id is the unique identifier for the object.

    Returns:
        np.ndarray: The frame with tracking bounding boxes and track IDs drawn.
    """
    for idx, (tid, x1, y1, x2, y2) in enumerate(tracks):
        # Adjust bounding box to be tighter
        x1, y1, x2, y2 = adjust_bbox(x1, y1, x2, y2)

        # Cycle through predefined colors for each track ID
        colour = COLOURS[idx % len(COLOURS)]

        # Draw the bounding box for the tracked object
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Display track ID near the top-left corner of the box
        cv2.putText(frame, f'ID:{tid}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    return frame
