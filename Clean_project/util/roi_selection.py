import cv2

def select_roi_from_video(video_path):
    """
    Allows the user to select a Region of Interest (ROI) from a video.

    Args:
        video_path (str): The path to the video file.

    Returns:
        list: A list of coordinates defining the ROI, or an empty list if no ROI is selected.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        raise ValueError("Error: Could not open video file.")

    # Create a window and display the video
    cv2.namedWindow("Select ROI")
    cv2.imshow("Select ROI", frame)

    # Initialize variables to store ROI coordinates
    roi = [-1, -1, -1, -1]
    roi_selected = False

    def select_roi(event, x, y, flags, param):
        nonlocal roi, roi_selected

        if event == cv2.EVENT_LBUTTONDOWN:
            roi[0], roi[1] = x, y
            roi_selected = False

        elif event == cv2.EVENT_LBUTTONUP:
            roi[2], roi[3] = x, y
            roi_selected = True
            # Draw a rectangle around the selected ROI
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
            cv2.imshow("Select ROI", frame)

    cv2.setMouseCallback("Select ROI", select_roi)

    while not roi_selected:
        if cv2.waitKey(100) & 0xFF == 27:  # Press Esc key to exit
            break

    # Close the video window
    cv2.destroyWindow("Select ROI")

    # Release the video capture object
    cap.release()

    if roi_selected:
        return roi
    else:
        return []
