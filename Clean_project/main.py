import cv2
import os
from util.video_io import process_video
from util.roi_selection import select_roi_from_video
from util.deep_sort_utils import initialize_deep_sort_tracker
from util.yolo import load_yolo_model, run_yolo_detections
from util.ftnet_loading import load_model
from util.feature_extraction import extract_feature_vectors
from util.image_transforming import processing_image


def main():
    # Initialize models and select ROI
    model_ft_net = load_model('model/ft_ResNet50/net_last.pth')
    yolo_model = load_yolo_model('model/yolo/yolov8n.pt')
    deep_sort_tracker = initialize_deep_sort_tracker()
    video_path = 'Videos/3_morning_cut.mp4'  # Replace with your video path
    roi = select_roi_from_video(video_path)

    # Process video
    process_video(video_path, roi, yolo_model, deep_sort_tracker)
    folder_path = '../gen_data_crop/3'  # Replace with your folder path
    save_path = '../generated_data/3_feature_vector'
    features, paths = extract_feature_vectors(folder_path, save_path, model_ft_net, processing_image)


if __name__ == "__main__":
    main()
