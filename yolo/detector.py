import os.path
import cv2
from yolo.utils import load_class_names
import os
from pathlib import Path

class YOLODetector:
    def __init__(self, config: str, weights: str, names: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> None:
        if not os.path.exists(config):
            print(f"{Path(__file__).parent}\\..\\{config}")
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.class_names = load_class_names(names)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detected_objects(self, image) -> list:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        if isinstance(unconnected_layers, tuple) or len(unconnected_layers.shape) == 1:
            output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
        else:
            output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
        outputs = self.net.forward(output_layers)

        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    detections.append({
                        "class_id": class_id,
                        "confidance": float(confidence),
                        "bbox": [x, y, int(w), int(h)]
                    })
        
        boxes = [d["bbox"] for d in detections]
        confidences = [d["confidance"] for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return []