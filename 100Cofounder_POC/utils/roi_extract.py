import cv2

# Global variables for mouse callback
roi = []
drawing = False
ix, iy = -1, -1

def select_roi(frame, window_name):
    """
    Allows the user to draw a rectangle on the frame and returns the coordinates.

    Parameters:
    frame (numpy.ndarray): The frame on which the user will draw the ROI.
    window_name (str): The name of the window in which the frame is displayed.

    Returns:
    list: The coordinates of the drawn rectangle in the format [x, y, width, height].
    """
    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, roi

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi = [ix, iy, x - ix, y - iy]
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)

    while True:
        if not drawing:
            cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break
        elif k == 13:  # Enter key to confirm ROI
            break

    cv2.destroyAllWindows()
    return roi if roi else None


def get_user_selected_roi(video_path, window_name):
    """
    Displays the first frame of the video and allows the user to select a Region of Interest (ROI).

    This function opens the video specified by the video_path, displays the first frame, and then 
    uses the select_roi function to allow the user to draw and select an ROI on this frame.

    Parameters:
    video_path (str): The path to the video file.
    window_name (str): The name of the window in which the video frame is displayed.

    Returns:
    list: The selected ROI in the format [x, y, width, height], or None if the selection is not made.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to read video frame from {video_path}")
        cap.release()
        return None

    selected_roi = select_roi(frame, window_name)  # Using the select_roi function from before
    cap.release()
    return selected_roi
