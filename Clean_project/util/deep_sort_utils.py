from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric


def initialize_deep_sort_tracker(max_age=10, n_init=5):
    """
    Initializes and returns a DeepSort tracker object with configurable parameters.

    Args:
        max_cosine_distance (float): Max cosine distance for the metric.
        nn_budget (int): Budget for nearest neighbor distance metric.
        max_iou_distance (float): Max IOU distance for matching.
        max_age (int): Maximum age of the tracker.
        n_init (int): Number of frames to initialize a track.

    Returns:
        DeepSort: An instance of the DeepSort tracker.
    """
    return DeepSort(max_age=max_age, n_init=n_init)


def prepare_deepsort_detections(detections, confidence_threshold):
    """
    Prepares detections for the DeepSort tracker.

    Args:
        detections (list): List of detections from a detection model.
        confidence_threshold (float): Threshold for filtering detections based on confidence.

    Returns:
        list: A list of detections formatted for DeepSort.
    """
    deepsort_detections = []
    for detection in detections.boxes.data.tolist():
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        if confidence < confidence_threshold:
            continue


        # Convert bounding box coordinates from (x_min, y_min, x_max, y_max) to (x, y, w, h)
        bbox = [int(x_min), int(y_min), int(x_max)-int(x_min), int(y_max)-int(y_min)]
        deepsort_detections.append((bbox, confidence, class_id))

    return deepsort_detections

