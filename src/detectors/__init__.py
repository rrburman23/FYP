"""
Detector package initialiser.
Re-exports the common detector classes so users can write:

    from detectors import YOLOv5, SSD, DetectorBase
"""

from .base import DetectorBase       
from .yolov5        import YOLOv5
from .fasterrcnn    import FasterRCNN
from .yolo          import YOLOv3
from .retinanet     import RetinaNet
from .ssd           import SSDDetector

__all__ = [
    "DetectorBase",
    "YOLOv5",
    "FasterRCNN",
    "YOLOv3",
    "RetinaNet",
    "SSDDetector",
]