# Multi-Camera Tracking and Reidentification Application

This application is designed to perform multi-camera tracking and reidentification using Python.

## Features

- Multi-camera support
- Region of Interest (ROI) selection for each camera feed
- Object detection and tracking
- Reidentification of tracked objects across different camera feeds
- Batch processing of images for reidentification

## Dependencies

- Python
- OpenCV
- PyTorch
- torchvision
- NumPy
- scikit-learn

## Setup

1. Clone the repository to your local machine.
2. Install the required dependencies.

## Usage

1. Run the `main.py` script to start the application.
2. For each camera feed, a window will open allowing you to select a Region of Interest (ROI).
3. The application will then start processing the video feeds, performing detection, tracking, and reidentification.

## Project Structure

- `main.py`: The main script to run the application.
- `utils/initialize.py`: Contains functions to initialize the reidentification model and start threads for each camera feed.
- `utils/camera_utils.py`: Contains functions for processing video frames, including detection, tracking, and saving cropped images.
- `utils/shared_data.py`: Contains shared data for the application, including a dictionary for track ID mapping and a lock for thread-safe access to the dictionary.
- `utils/reid_helper.py`: Contains helper functions for the reidentification process.
- `utils/detection_tracking.py`: Contains the `DetectionTracking` class for object detection and tracking.
- `models/ft_ResNet50/model.py`: Contains the `ft_net` class for the reidentification model.

## Contributing

Contributions are welcome. Please open an issue to discuss your ideas or submit a pull request with your changes.

## License

This project is licensed under the terms of the MIT license.
