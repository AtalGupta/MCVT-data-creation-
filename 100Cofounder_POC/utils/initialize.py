import threading
from utils.camera_utils import display_camera_feed
from utils.reid_helper import GALLERY_FOLDER_PATH, FILENAME_MAPPING_PATH, FEATURE_FILE_PATH, initialize_gallery
import torch
from models.ft_ResNet50.model import ft_net
from utils.roi_extract import get_user_selected_roi

def initialize_reid_model():
    """
    Initializes the re-identification model and gallery.
    Returns the initialized model.
    """
    model_ft_net = ft_net(class_num=751)
    state_dict = torch.load('models/ft_ResNet50/net_last.pth', map_location=torch.device('cpu'))
    model_ft_net.load_state_dict(state_dict)
    model_ft_net.eval()
    initialize_gallery(GALLERY_FOLDER_PATH, model_ft_net, FEATURE_FILE_PATH, FILENAME_MAPPING_PATH)
    return model_ft_net

def start_camera_threads(video_paths, model):
    """
    Starts threads for processing each camera feed.

    This function iterates over the provided video paths, calls get_user_selected_roi to let the user 
    select an ROI for each video feed, and then starts a thread for display_camera_feed function with 
    the selected ROI.

    Parameters:
    video_paths (list of tuples): A list where each tuple contains a video path and a camera number.

    Returns:
    list: A list of threading.Thread objects representing the started threads for each camera feed.
    """
    threads = []
    for video_path, camera_no in video_paths:
        window_name = f"Camera {camera_no}"
        roi = get_user_selected_roi(video_path, window_name)
        if roi is None:
            print(f"No ROI selected for {window_name}, skipping this camera.")
            continue  # Skip this camera if ROI was not selected
        thread = threading.Thread(target=display_camera_feed, args=(video_path, window_name, camera_no, roi, model))
        threads.append(thread)
        thread.start()
    return threads
