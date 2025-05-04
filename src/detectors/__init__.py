"""
Detector package initialiser.
Re-exports the common detector classes so users can write:
"""

from .base          import DetectorBase       
from .yolov5        import YOLOv5
from .fasterrcnn    import FasterRCNN
from .yolo          import YOLOv3
from .ssd           import SSDDetector

__all__ = [
    "DetectorBase",
    "YOLOv5",
    "FasterRCNN",
    "YOLOv3",
    "SSDDetector"
]