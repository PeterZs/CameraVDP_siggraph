# World coordinates are represented in pixels instead of meters
import numpy as np
import cv2 as cv
import argparse
from screeninfo import get_monitors


def get_second_screen_resolution_and_position():
    """
    Retrieve the resolution and position of the second monitor.
    If only one monitor is available, return the primary monitor's resolution and position.

    Returns:
        tuple: (width, height, x, y)
    """
    monitors = get_monitors()
    monitor = monitors[1] if len(monitors) > 1 else monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y


def create_checkerboard(square_size=50,
                        rows=6,
                        cols=7,
                        background_size=(800, 800),
                        background_color=(128, 128, 128)):
    """
    Create a centered checkerboard pattern over a solid background.

    Args:
        square_size (int): Size of each square in pixels (both width and height).
        rows (int): Number of checkerboard rows.
        cols (int): Number of checkerboard columns.
        background_size (tuple): Background image size (width, height).
        background_color (tuple): Background color in BGR format.

    Returns:
        numpy.ndarray: Checkerboard image with background.
    """
    # Dimensions of the checkerboard
    board_width = cols * square_size
    board_height = rows * square_size

    # Initialize background
    background = np.ones((background_size[1], background_size[0], 3), dtype=np.uint8) \
                 * np.array(background_color, dtype=np.uint8)

    # Create checkerboard (initialized as white)
    checkerboard = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:  # Alternate black squares
                top_left = (j * square_size, i * square_size)
                bottom_right = ((j + 1) * square_size - 1, (i + 1) * square_size - 1)
                cv.rectangle(checkerboard, top_left, bottom_right, (0, 0, 0), -1)

    # Center the checkerboard on the background
    start_x = (background_size[0] - board_width) // 2
    start_y = (background_size[1] - board_height) // 2
    background[start_y:start_y + board_height, start_x:start_x + board_width] = checkerboard

    return background


def main():
    # ---------------------- Argument Parsing ---------------------- #
    parser = argparse.ArgumentParser(description="Generate a fullscreen checkerboard pattern.")
    parser.add_argument("--square_size", type=int, default=64,
                        help="Size of each square in pixels (default: 64)")
    parser.add_argument("--rows", type=int, default=32,
                        help="Number of checkerboard rows (default: 32)")
    parser.add_argument("--cols", type=int, default=58,
                        help="Number of checkerboard columns (default: 58)")
    parser.add_argument("--background_color", type=int, nargs=3, default=[255, 255, 255],
                        help="Background color in BGR format (default: 255 255 255)")

    args = parser.parse_args()
    if args.rows % 2 != 0 or args.cols % 2 != 0:
        raise ValueError("Both rows and cols must be even numbers.")

    # ---------------------- Execution ---------------------- #
    # Get second screen resolution
    screen_width, screen_height, screen_x, screen_y = get_second_screen_resolution_and_position()
    background_size = (screen_width, screen_height)

    # Generate checkerboard with background
    checkerboard_image = create_checkerboard(square_size=args.square_size,
                                             rows=args.rows,
                                             cols=args.cols,
                                             background_size=background_size,
                                             background_color=tuple(args.background_color))

    # Display fullscreen checkerboard on second monitor
    window_name = "Checker Board"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, screen_x, screen_y)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(window_name, checkerboard_image)

    # Optional: save to file
    # cv.imwrite("checkerboard_fullscreen.jpg", checkerboard_image)

    # Wait for key press and close window
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
