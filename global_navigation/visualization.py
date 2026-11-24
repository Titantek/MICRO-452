"""Show the map and paths on the image from computer vision."""

import cv2
import numpy as np

from computer_vision.visualization import draw_obstacles, draw_pose


def draw_navigation(frame, plan=None, grid=None):
    """Orange = clearance, blue = A*, green = final path. Colours are BGR."""
    image = frame.warped.copy()
    if plan is not None:
        grid = plan.grid

    if grid is not None:
        height, width = image.shape[:2]
        mask = cv2.resize(
            grid.occupied.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        image[mask] = (0.65 * image[mask] + 0.35 * np.array([0, 165, 255])).astype(np.uint8)

    draw_obstacles(image, frame.obstacles, labels=False)
    if plan is not None:
        for path, color in ((plan.raw_path_m, (255, 100, 0)), (plan.waypoints, (0, 180, 0))):
            pixels = np.rint(grid.meters_to_pixels(path)).astype(np.int32)
            cv2.polylines(image, [pixels], False, color, 3, cv2.LINE_AA)

    draw_pose(image, frame.thymio, "thymio", (255, 0, 0))
    draw_pose(image, frame.goal, "goal", (0, 160, 0))
    return image
