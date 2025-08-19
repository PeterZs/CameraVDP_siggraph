import cv2
import numpy as np
from screeninfo import get_monitors

def get_second_screen_resolution_and_position():
    """
    Get the resolution and position of the second monitor.
    If only one monitor is available, return the primary monitor's resolution and position.
    :return: (width, height, x, y)
    """
    monitors = get_monitors()
    if len(monitors) > 1:
        monitor = monitors[1]
    else:
        monitor = monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y

# Get the resolution and position of the second monitor
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# Set rectangle parameters
resize_factor = 1
rect_width = round(screen_width * resize_factor)  # Rectangle width (pixels)
rect_height = round(screen_height * resize_factor)  # Rectangle height (pixels)
rect_color = (255, 255, 255)  # Rectangle color (white)

# Create a black background
background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

# Calculate the rectangle position to center it
rect_x = (screen_width - rect_width) // 2
rect_y = (screen_height - rect_height) // 2

# Draw the rectangle on the background
cv2.rectangle(background, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rect_color, -1)

# Create a window
window_name = "Central Rectangle"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # Move the window to the second monitor
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Display the image
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
