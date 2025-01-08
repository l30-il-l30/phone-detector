import cv2

def load_class_names(names_file: str) -> list[str]:
    with open(names_file, "r") as f:
        return [line.strip() for line in f.readlines()]

def draw_bounding_boxes(image: any, detections: any, class_names: any) -> any:
    for detection in detections:
        x, y, w, h = detection["bbox"]
        class_id = detection["class_id"]
        label = class_names[class_id]
        confidence = detection["confidance"]

        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image
