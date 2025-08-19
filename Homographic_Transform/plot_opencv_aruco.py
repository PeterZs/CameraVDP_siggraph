import cv2
import numpy as np
import os
from screeninfo import get_monitors

def get_second_screen_resolution_and_position():
    """
    Get the resolution and position of the second display.
    If there is only one display, return the primary display's resolution and position.
    :return: (width, height, x, y)
    """
    monitors = get_monitors()
    if len(monitors) > 1:
        monitor = monitors[1]
    else:
        monitor = monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y

# Set the ArUco marker dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# Display parameters
display_width_pixel = 3840   # Display width (pixels)
display_height_pixel = 2160  # Display height (pixels)

# Marker parameters
marker_size = 48
spacing_x = 48
spacing_y = 48

num_rows = 22  # Number of rows
num_cols = 40  # Number of columns

# Get the second display's resolution and position
screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()

# Create a white background matching the second screen resolution
background = np.ones((screen_height, screen_width), dtype=np.uint8) * 255

# Compute total grid width and height
grid_width = num_cols * marker_size + (num_cols - 1) * spacing_x
grid_height = num_rows * marker_size + (num_rows - 1) * spacing_y

x_bias = 0
y_bias = 0

# Ensure the grid is centered
x_start = (screen_width - grid_width) // 2 + x_bias
y_start = (screen_height - grid_height) // 2 - y_bias

# Generate and place ArUco markers
for row in range(num_rows):
    for col in range(num_cols):
        # Compute the current marker index
        marker_id = row * num_cols + col

        # Generate the ArUco marker
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

        # Compute the marker placement position
        x_offset = x_start + col * (marker_size + spacing_x)
        y_offset = y_start + row * (marker_size + spacing_y)

        # Place the ArUco marker onto the background
        background[y_offset:y_offset + marker_size, x_offset:x_offset + marker_size] = marker_image

# Create window
window_name = "Aruco Grid"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, screen_x, screen_y)  # Move the window to the second screen
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Display the image
cv2.imshow(window_name, background)
cv2.waitKey(0)
cv2.destroyAllWindows()
