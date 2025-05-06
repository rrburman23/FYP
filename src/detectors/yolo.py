"""
YOLOv3 detector implementation using OpenCV DNN backend.

This module defines the YOLOv3 class, which inherits from DetectorBase and implements
object detection using the YOLOv3 model. It integrates with getModels.py to automatically
download model files if they are missing.
"""

import cv2
import numpy as np
import logging
from .base import DetectorBase
from util import ROOT_DIR, get_models

# Configure logging for debugging and monitoring
logger = logging.getLogger(__name__)

class YOLOv3(DetectorBase):
    def __init__(self, model: str = "yolov3", conf_thresh: float = 0.4, nms_thresh: float = 0.45):
        """
        Initialize YOLOv3 detector using OpenCV DNN.

        Automatically downloads model files (config, weights, labels) if missing using
        getModels.py. Loads the YOLOv3 model and prepares it for inference.

        :param model: Model name (e.g., 'yolov3'), used to identify config files
        :param conf_thresh: Confidence threshold for detections
        :param nms_thresh: Non-max suppression threshold
        :raises FileNotFoundError: If class names file cannot be loaded
        :raises RuntimeError: If model loading fails
        """
        self.confidence_threshold = conf_thresh
        self.nms_threshold = nms_thresh

        # Download model files if they don't exist
        try:
            logger.info("Checking for YOLOv3 model files...")
            get_models()
            logger.info("Model files check completed")
        except Exception as e:
            logger.error(f"Failed to download model files: {e}")
            raise RuntimeError(f"Model file download failed: {e}")

        # Paths to model files (relative to ROOT_DIR, as specified in getModels.py)
        cfg_path = ROOT_DIR / "models/yolov3.cfg"
        weights_path = ROOT_DIR / "models/yolov3.weights"
        names_path = ROOT_DIR / "models/labels.names"

        # Load class names
        try:
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.debug(f"Loaded {len(self.classes)} class names from {names_path}")
        except FileNotFoundError:
            logger.error(f"Class names file not found at {names_path}")
            raise FileNotFoundError(f"Class names file not found at {names_path}")

        # Load YOLO model
        try:
            self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info(f"Successfully loaded YOLOv3 model from {cfg_path} and {weights_path}")
        except cv2.error as e:
            logger.error(f"Failed to load YOLOv3 model: {e}")
            raise RuntimeError(f"Failed to load YOLOv3 model from {cfg_path} and {weights_path}: {e}")

        # Get output layer names for inference
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        logger.debug(f"Initialized output layers: {self.output_layers}")

    def detect(self, frame: cv2.Mat) -> np.ndarray:
        """
        Detect objects in the input frame using YOLOv3.

        Processes the input frame, runs inference, and applies non-max suppression to
        filter detections.

        :param frame: Input image (BGR format)
        :return: Array of detections [x1, y1, x2, y2, confidence]
        """
        height, width = frame.shape[:2]

        # Preprocess frame for YOLO input
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Perform forward pass
        outputs = self.net.forward(self.output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # Scale bounding box coordinates to image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x1 = int(center_x - w/2)
                    y1 = int(center_y - h/2)

                    boxes.append([x1, y1, x1 + w, y1 + h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        # Format detections for output
        detections = []
        for i in indices:
            box = boxes[i]
            detections.append([box[0], box[1], box[2], box[3], confidences[i]])

        # Return detections as a NumPy array
        return np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)