import cv2
from config import MODEL_CONFIG, MODEL_WEIGHT, CLASS_NAMES, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
from yolo.detector import YOLODetector
from yolo.utils import draw_bounding_boxes

def main() -> None:
    yolo = YOLODetector(config=MODEL_CONFIG, weights=MODEL_WEIGHT, names=CLASS_NAMES, conf_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        detections = yolo.detected_objects(frame)
        cell_phones = [d for d in detections if d["class_id"] == 67]
        frame = draw_bounding_boxes(frame, cell_phones, yolo.class_names)
        cv2.putText(frame, f"Cell Phones: {len(cell_phones)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
