from __future__ import annotations
from typing import List
import torch
from torchvision import models, transforms as T
import numpy as np
from detectors.base import DetectorBase

class FasterRCNN(DetectorBase):
    """
    A wrapper class for the Faster R-CNN object detection model provided by 
    torchvision. This class facilitates object detection using a pre-trained 
    Faster R-CNN model (ResNet50 backbone with FPN) on input images.

    Attributes:
        device (str): The device on which the model will run, either 'cpu' or 'cuda'.
        model (torch.nn.Module): The Faster R-CNN model loaded onto the specified device.
        tf (torchvision.transforms.Compose): The transformation pipeline for input frame.
        th (float): The confidence threshold for detections; only detections with a 
                    confidence score higher than this value are returned.
    """

    def __init__(self, model: str = "fasterrcnn_resnet50_fpn", conf_thresh: float = 0.4, device: str | None = None):
        """
        Initialize the FasterRCNN object detector.

        Args:
            conf_thresh (float): Minimum confidence threshold to keep a detection. 
                                  Defaults to 0.4.
            device (str | None): The device to use for inference. If None, the 
                                 function automatically chooses 'cuda' if available, 
                                 otherwise defaults to 'cpu'.
        """
        if model != "fasterrcnn_resnet50_fpn":
            raise ValueError(f"Unsupported FasterRCNN model: {model}")
        
        # Set the device for inference (either 'cuda' or 'cpu')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained Faster R-CNN model with a ResNet-50 FPN backbone
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(self.device).eval()

        # Define the transformation pipeline to convert input frame to a PyTorch tensor
        self.tf = T.Compose([T.ToTensor()])

        # Set the confidence threshold for filtering low-confidence detections
        self.th = conf_thresh

    @torch.inference_mode()
    def detect(self, frame) -> List[list]:
        """
        Perform object detection on a given frame using the Faster R-CNN model.

        The method preprocesses the input frame, performs the object detection, 
        and returns the bounding boxes of detected objects along with their 
        confidence scores.

        Args:
            frame (np.ndarray): The input image (frame) on which to perform object detection.
                                 The frame should be in BGR format (as used by OpenCV).

        Returns:
            List[list]: A list of detected objects in the form [x1, y1, x2, y2, confidence],
                        where (x1, y1) and (x2, y2) are the coordinates of the bounding box, 
                        and `confidence` is the confidence score of the detection.
        """
        # Ensure that the frame is a contiguous array to handle memory layout issues
        # The frame is initially in BGR format (OpenCV default), so we convert it to RGB
        frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB + ensures contiguous memory layout
        
        # Apply transformation to convert the frame to a tensor and add a batch dimension
        transformed_frame = self.tf(frame_rgb).unsqueeze(0).to(self.device)

        # Perform detection without tracking gradients
        with torch.no_grad():
            # Get model's predictions for the transformed frame
            out = self.model(transformed_frame)[0]

        # Extract bounding boxes and confidence scores from the model's output
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        # Filter and return only the detections that have a confidence score above the threshold
        return [
            [int(x1), int(y1), int(x2), int(y2), float(s)]
            for (x1, y1, x2, y2), s in zip(boxes, scores) 
            if s > self.th  # Only keep detections with confidence > threshold
        ]
