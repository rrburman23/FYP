from typing import List
import torch
import numpy as np
from torchvision import transforms as T, models

class SSDDetector:
    """
    Single-Shot Detector (SSD-300 VGG-16 backbone) using torchvision.
    This class wraps the SSD detector and exposes a uniform `.detect(frame)` API.
    """

    def __init__(self, conf_thresh: float = 0.5, device: str | None = None):
        """
        Initialize SSD detector with pretrained weights.

        Args:
            conf_thresh (float): Minimum confidence threshold to keep a detection.
            device (str | None): The device to run the model on ("cuda" for GPU or "cpu" for CPU).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.detection.ssd300_vgg16(weights="DEFAULT").to(self.device)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
        self.conf_thresh = conf_thresh

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> List[list]:
        """
        Detect objects using SSD-300.

        Args:
            frame (np.ndarray): Input frame to perform object detection.

        Returns:
            List[list]: List of bounding boxes and confidence scores in the format [x1, y1, x2, y2, confidence].
        """
        tensor = self.transform(frame[:, :, ::-1].copy()).to(self.device)
        output = self.model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        results = []
        for bbox, score in zip(boxes, scores):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            results.append([x1, y1, x2, y2, float(score)])

        return results
