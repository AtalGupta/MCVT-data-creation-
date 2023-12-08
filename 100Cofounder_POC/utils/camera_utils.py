import cv2
from utils.detection_tracking import DetectionTracking
import os
from utils.reid_helper import extract_save_feature_dir
from utils.reid_helper import extract_and_update_gallery_features, calculate_similarity_matrix, assign_track_ids_and_update_gallery
from utils.shared_data import track_id_mapping, track_id_mapping_lock

TEMP_GALLERY_FOLDER = "temporary_gallery"
MAIN_GALLERY_FOLDER = "gallery"

def update_track_id_mapping(original_id, updated_id):
    """
    Updates the track ID mapping dictionary with a new mapping.

    Parameters:
    original_id (str): The original track ID.
    updated_id (str): The updated track ID.
    """
    with track_id_mapping_lock:
        track_id_mapping[original_id] = updated_id

def get_updated_track_id(original_id):
    """
    Retrieves the updated track ID from the track ID mapping dictionary.

    Parameters:
    original_id (str): The original track ID.

    Returns:
    str: The updated track ID if it exists, otherwise the original ID.
    """
    with track_id_mapping_lock:
        return track_id_mapping.get(original_id, original_id)

def select_roi(video_path):
    """
    Allows the user to select a region of interest (ROI) on the video feed.

    Parameters:
    video_path (str): Path to the video file.

    Returns:
    tuple: Coordinates of the selected ROI (x, y, w, h).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading video file: {video_path}")
        return None

    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    cap.release()
    return roi

def draw_bounding_box(frame, bbox, track_id):
    """
    Draws a bounding box and track ID on the frame.

    Parameters:
    frame (np.ndarray): The frame to draw on.
    bbox (tuple): The bounding box coordinates (x1, y1, x2, y2).
    track_id (str): The track ID.
    """
    print(f"Track ID: {track_id}, BBox: {bbox}")
    if track_id is not None:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

BATCH_SIZE = 64  # Number of images per batch
current_batch = 0  # Current batch number
image_counter = 0  # Counter for images in the current batch

def process_video_frame(frame, detect_track, roi, frame_counter, camera_no, model):
    """
    Processes a video frame, performs detection and tracking, and saves cropped images.

    Parameters:
    frame (np.ndarray): The frame to process.
    detect_track (DetectionTracking): The detection and tracking object.
    roi (tuple): The region of interest.
    frame_counter (int): The current frame number.
    camera_no (int): The camera number.
    model (torch model): The reidentification model.

    Returns:
    np.ndarray: The processed frame.
    list: The list of cropped images.
    """
    global current_batch, image_counter
    x, y, w, h = roi
    cropped_images = []

    frame_for_cropping = frame.copy()

    tracks = detect_track.process_frame(frame)
    for track in tracks:
        bbox = track.to_tlbr()
        track_id = track.track_id
        adjusted_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])

        center_x, center_y = (adjusted_bbox[0] + adjusted_bbox[2]) / 2, (adjusted_bbox[1] + adjusted_bbox[3]) / 2
        if x <= center_x <= x + w and y <= center_y <= y + h:
            if frame_counter % 5 == 0:
                cropped_img = frame_for_cropping[int(adjusted_bbox[1]):int(adjusted_bbox[3]),
                              int(adjusted_bbox[0]):int(adjusted_bbox[2])]
                cropped_images.append(cropped_img)

                gallery_folder = MAIN_GALLERY_FOLDER if camera_no == 0 else TEMP_GALLERY_FOLDER
                batch_folder = os.path.join(gallery_folder, f"batch_{current_batch}")
                os.makedirs(batch_folder, exist_ok=True)

                filename = os.path.join(batch_folder, f"{track_id}_{camera_no}_{frame_counter}.jpg")
                cv2.imwrite(filename, cropped_img)

                if camera_no != 0:
                    image_counter += 1
                    if image_counter >= BATCH_SIZE:
                        batch_feature_folder = os.path.join(gallery_folder, f"batch_{current_batch}_feature")
                        os.makedirs(batch_feature_folder, exist_ok=True)
                        extract_save_feature_dir(batch_folder, model, batch_feature_folder)
                        extract_and_update_gallery_features(MAIN_GALLERY_FOLDER, model, 'gallery_features')
                        similarity_matrix, batch_filenames, gallery_filenames = calculate_similarity_matrix(
                            batch_feature_folder, 'gallery_features')
                        assign_track_ids_and_update_gallery(similarity_matrix, batch_filenames, gallery_filenames,
                                                            batch_folder, MAIN_GALLERY_FOLDER)

                        current_batch += 1
                        image_counter = 0

        draw_bounding_box(frame, adjusted_bbox, track_id)

    return frame, cropped_images

def display_camera_feed(video_path, window_name, camera_no, roi, model):
    """
    Displays the camera feed with detections and tracking.

    Parameters:
    video_path (str): Path to the video file.
    window_name (str): Name of the window to display the feed.
    camera_no (int): The camera number.
    roi (tuple): The region of interest.
    model (torch model): The reidentification model.
    """
    cap = cv2.VideoCapture(video_path)
    detect_track = DetectionTracking()
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, cropped_images = process_video_frame(frame, detect_track, roi, frame_counter, camera_no, model)
        frame_counter += 1
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()