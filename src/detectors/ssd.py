"""
ssd.py
======
Single‑Shot Detector (SSD‑300 VGG‑16 backbone) wrapper using torchvision.

Loads a pretrained network and exposes a uniform `.detect(frame)` API that
returns `[x1, y1, x2, y2, confidence]` for every detection above a threshold.
"""

from typing import List
import torch
import numpy as np
from torchvision import transforms as T, models

# ──────────────────────────────────────────────────────────────────────────────
class SSDDetector:
    """Pre‑trained SSD‑300 detector."""

    def __init__(self,
                 conf_thresh: float = 0.5,
                 device: str | None = None) -> None:
        """
        Args:
            conf_thresh (float): Minimum confidence score to keep a detection.
            device (str | None): "cuda" or "cpu". Defaults to auto‑detect.
        """
        # Auto-select device if none specified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained SSD model (SSD300 with VGG16 backbone)
        self.model = models.detection.ssd300_vgg16(weights="DEFAULT").to(self.device)
        
        # Set the model to evaluation mode for inference
        self.model.eval()
        
        # Define the transform to convert images to tensors
        self.transform = T.Compose([T.ToTensor()])
        
        # Set the confidence threshold for keeping detections
        self.conf_thresh = conf_thresh

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> List[list]:
        """
        Run object detection on a single BGR frame.

        Args:
            frame (np.ndarray): OpenCV image (H x W x 3, BGR).

        Returns:
            list[list]: [[x1, y1, x2, y2, confidence], …] in pixel coords.
        """
        # Convert BGR image to RGB and normalize to range [0, 1]
        tensor = self.transform(frame[:, :, ::-1].copy()).to(self.device)
        
        # Perform inference on a batch of size 1 (frame)
        output = self.model([tensor])[0]
        
        # Extract bounding boxes and scores from the output
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        # Filter detections based on confidence threshold
        results = []
        for bbox, score in zip(boxes, scores):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            results.append([x1, y1, x2, y2, float(score)])

        # Debugging: Log detection results
        print(f"[SSD] frame detections: {len(results)}")  # Log number of detections
        for det in results:  # Log each detection's bounding box and score
            print(f"Detection: {det}")

        return results