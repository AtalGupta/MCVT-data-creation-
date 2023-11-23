from ultralytics import YOLO

def load_yolo_model(model_path):
    """
    Loads the YOLO model from the given path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        YOLO: An instance of the YOLO model.
    """
    return YOLO(model_path)

def run_yolo_detections(model, frame, classes=[0]):
    """
    Runs YOLO detections on a given frame with a configurable confidence threshold.

    Args:
        model (YOLO): The YOLO model instance.
        frame (numpy.ndarray): The frame on which to run detections.
        confidence_threshold (float): Threshold for filtering detections based on confidence.
        classes (list, optional): List of classes to detect. Defaults to [0].

    Returns:
        list: A list of detection results.
    """
    return model(frame, classes=classes)[0]
