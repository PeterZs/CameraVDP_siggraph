import argparse
import cv2 as cv
import numpy as np
from screeninfo import get_monitors

# Color list for display (kept as-is to preserve visual output behavior).
# Note: Although the comment in the original code mentioned RGB→BGR conversion,
# the original code displayed these tuples directly with OpenCV.
# To keep the visual result unchanged, we also pass them directly to OpenCV.
colorchecker_colors = [
    (68, 82, 115), (130, 150, 194), (157, 122, 98), (67, 108, 87),
    (177, 128, 133), (170, 189, 103), (44, 126, 214), (166, 91, 80),
    (99, 90, 193), (108, 60, 94), (64, 188, 157), (46, 163, 224),
    (150, 61, 56), (73, 148, 70), (60, 54, 175), (31, 199, 231),
    (149, 86, 187), (158, 166, 50), (242, 243, 242), (200, 200, 200),
    (160, 160, 160), (121, 122, 122), (85, 85, 85), (52, 52, 52),
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
]


def get_second_screen_resolution_and_position():
    """
    Get the resolution and top-left position of the second monitor.
    If only one monitor is present, return the primary monitor.

    Returns:
        tuple: (width, height, x, y)
    """
    monitors = get_monitors()
    monitor = monitors[1] if len(monitors) > 1 else monitors[0]
    return monitor.width, monitor.height, monitor.x, monitor.y


def main():
    # ---------------------- Argument Parsing ---------------------- #
    parser = argparse.ArgumentParser(
        description="Display a fullscreen rectangle cycling through ColorChecker-like colors on the second monitor."
    )
    parser.add_argument(
        "--rect_width", type=int, default=2000,
        help="Rectangle width in pixels (default: 2000)"
    )
    parser.add_argument(
        "--rect_height", type=int, default=2000,
        help="Rectangle height in pixels (default: 2000)"
    )
    args = parser.parse_args()

    # ---------------------- Screen Setup -------------------------- #
    screen_w, screen_h, screen_x, screen_y = get_second_screen_resolution_and_position()

    # Compute centered rectangle position
    rect_w = args.rect_width
    rect_h = args.rect_height
    rect_x = (screen_w - rect_w) // 2
    rect_y = (screen_h - rect_h) // 2

    # ---------------------- Window Setup -------------------------- #
    window_name = "ColorChecker Display"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.moveWindow(window_name, screen_x, screen_y)            # Move to second monitor
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # ---------------------- Display Loop -------------------------- #
    color_index = 0
    while True:
        # Black background
        background = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        print(f"Showing Color {color_index}")

        # Draw centered rectangle with current color
        rect_color = colorchecker_colors[color_index]
        cv.rectangle(background, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), rect_color, -1)

        # Show image
        cv.imshow(window_name, background)

        # Wait for key: Space → next color, ESC → exit
        key = cv.waitKey(0)
        if key == 32:        # Space
            color_index = (color_index + 1) % len(colorchecker_colors)
        elif key == 27:      # ESC
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    # World coordinates are in pixels
    main()
