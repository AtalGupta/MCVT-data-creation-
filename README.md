# Multi-Camera Tracking and Reidentification System

## Project Description
This project is designed for detecting, tracking, and reidentifying individuals across multiple CCTV camera feeds. It combines object detection, tracking algorithms, and feature extraction to analyze video data from various camera angles, creating a unified gallery of identified individuals.

## Project Structure

### Main Script
- `main.py`: Orchestrates video feed analysis, including ROI selection, tracking, feature extraction, and gallery creation.

### Utility Modules
- `util/video_io.py`: Manages video I/O operations.
- `util/roi_selection.py`: Facilitates ROI selection from video feeds.
- `util/deep_sort_utils.py`: Integrates DeepSort tracking functionalities.
- `util/yolo_utils.py`: Handles operations related to the YOLO model.
- `util/feature_extraction.py`: Dedicated to feature extraction from tracked images.

### Models Directory
- `models/`: Stores pre-trained models for object detection and feature extraction.

### Gallery Creation Script
- `create_gallery.py`: Script for creating a gallery from camera feeds.

## Installation
Install the necessary dependencies:
```bash
pip install -r requirements.txt
