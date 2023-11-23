import cv2
import os

from util.deep_sort_utils import prepare_deepsort_detections
from util.yolo import run_yolo_detections
def save_cropped_image(image, path, filename):
    """
    Saves a cropped image to the specified path.

    Args:
        image (numpy.ndarray): The image to be saved.
        path (str): The directory where the image will be saved.
        filename (str): The name of the file.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, filename), image)


def process_video(video_path, roi, yolo_model, deep_sort_tracker):
    """
    Processes the video: performs detection, tracking, and saves clean cropped images without drawn rectangles.

    Args:
        video_path (str): Path to the video file.
        roi (list): Region of Interest coordinates.
        yolo_model (YOLO): Loaded YOLO model for object detection.
        deep_sort_tracker (DeepSort): Initialized DeepSort tracker.
    """
    video_cap = cv2.VideoCapture(video_path)
    frame_no = 0
    CONFIDENCE_THRESHOLD = 0.4
    SAVE_FRAME_GAP = 20
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)
    CAMERA_NO = 3

    while True:
        ret, frame = video_cap.read()
        frame_no += 1
        if not ret:
            break

        # Make a copy of the frame for cropping
        frame_for_cropping = frame.copy()

        # Run YOLO detection
        detections = run_yolo_detections(yolo_model, frame)

        # for det in detections.boxes.data.tolist():
        #     x_min, y_min, x_max, y_max, confidence, _ = det
        #     if confidence >= CONFIDENCE_THRESHOLD:
        #         cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)



        # Prepare detections for deep SORT
        deepsort_detections = prepare_deepsort_detections(detections, CONFIDENCE_THRESHOLD)

        # Update the deep SORT tracker
        tracks = deep_sort_tracker.update_tracks(deepsort_detections, frame=frame)

        # Draw bounding boxes and track IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)

            # Check if center of bbox is inside ROI
            if roi[0] < bbox_center[0] < roi[2] and roi[1] < bbox_center[1] < roi[3]:
                if frame_no % SAVE_FRAME_GAP == 0:  # Save every specified frame
                    cropped_img = frame_for_cropping[ymin:ymax, xmin:xmax]
                    save_cropped_image(cropped_img, '../gen_data_crop/3', f"{track_id}_{CAMERA_NO}_{frame_no}.jpg")

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, str(track_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()
