import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class DetectionTracking:
    def __init__(self):
        self.model = YOLO("models/yolov8m.pt")
        self.tracker = DeepSort(max_age=10, n_init=5)

    def process_frame(self, frame):
        CONFIDENCE_THRESHOLD = 0.4  # Increased confidence threshold
        detections = self.model(frame, classes=[0])[0]

        deepsort_detections = []
        for data in detections.boxes.data.tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = data
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            bbox = [int(x_min), int(y_min), int(x_max) - int(x_min), int(y_max) - int(y_min)]
            deepsort_detections.append((bbox, confidence, int(class_id)))

        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        return tracks



