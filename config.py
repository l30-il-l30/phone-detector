# config.py

# https://github.com/AlexeyAB/darknet?tab=readme-ov-file
# https://github.com/AlexeyAB/darknet/blob/master/data/coco.names

# path to YOLO model files
MODEL_CONFIG: str = "models/yolov4.cfg"
MODEL_WEIGHT: str = "models/yolov4.weights"
CLASS_NAMES: str = "models/coco.names"

# Confidence threshold and NMS threshold
CONFIDENCE_THRESHOLD: float = 0.5
NMS_THRESHOLD: float = 0.4