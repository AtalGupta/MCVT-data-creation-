from utils.initialize import start_camera_threads
from utils.reid_helper import load_model



def main():
    """
    Main function to run the multi-camera tracking and reidentification application.
    """
    # Uncomment below line to initialize the re-identification model
    model_ft_net = load_model('models/ft_ResNet50/net_last.pth')

    # Define camera feeds
    video_paths = [('Videos/4_POC_30min.mp4', 0),
                   ('Videos/5_POC_30min.mp4', 1),
                   ('Videos/10_POC_30min.mp4', 2),
                   ]

    # Start and manage threads for camera feeds
    threads = start_camera_threads(video_paths, model_ft_net)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
