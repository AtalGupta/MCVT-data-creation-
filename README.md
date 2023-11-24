Multi-Camera Tracking and Reidentification System
Project Description
This project implements a multi-camera tracking and reidentification system, designed to detect, track, and reidentify individuals across multiple CCTV camera feeds. It combines object detection, tracking algorithms, and feature extraction to analyze video data from different camera angles and create a unified gallery of identified individuals.

Project Structure
main.py
The entry point of the application.
Orchestrates the process of video feed analysis, including ROI selection, tracking, feature extraction, and gallery creation.
utils/
A directory containing utility modules for various functionalities of the system.

video_io.py
Manages video input and output operations.
Functions for reading videos, saving cropped images, and video stream handling.
roi_selection.py
Facilitates the selection of Regions of Interest (ROI) from video feeds.
Essential for focusing the tracking on specific areas in video frames.
deep_sort_utils.py
Integrates DeepSort tracking functionalities.
Includes initialization of the DeepSort tracker and formatting detections for tracking.
yolo_utils.py
Handles the operations related to the YOLO model.
Responsible for loading the YOLO model and performing object detection.
feature_extraction.py
Dedicated to feature extraction from the tracked images.
Uses the ft_ResNet50 model for feature vector extraction, crucial in the re-identification process.
models/
A directory to store pre-trained models used for object detection and feature extraction.
create_gallery.py
A standalone script for creating a gallery from camera feeds.
Compares features from two different cameras, assigns track IDs, and organizes images into a gallery.
Installation
To install the necessary dependencies for this project, run:

bash
Copy code
pip install -r requirements.txt
Usage
Execute the main script to start processing the video feeds:

bash
Copy code
python main.py
Customize the script parameters as per your camera setup and requirements.
