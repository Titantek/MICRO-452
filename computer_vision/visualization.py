"""Drawing helpers for the top view (BGR images, OpenCV)."""

import cv2
import numpy as np

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 160, 0)
ORANGE = (0, 200, 255)


def draw_label(image, text, anchor, align_right=False, below=True, color=RED):
    """Text on a white box, kept inside the image."""
    font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 6
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = anchor[0] - text_w - 2 * pad if align_right else anchor[0]
    y = anchor[1] + text_h + 2 * pad if below else anchor[1] - baseline - 2 * pad
    height, width = image.shape[:2]
    x = int(min(max(x, pad), width - text_w - pad - 1))
    y = int(min(max(y, text_h + pad), height - baseline - pad - 1))
    cv2.rectangle(image, (x - pad, y - text_h - pad), (x + text_w + pad, y + baseline + pad),
                  (255, 255, 255), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_pose(image, pose, label, color, arrow_length=80):
    """Dot + heading arrow + label for an (x, y, theta) pose. Does nothing if pose is None."""
    if pose is None:
        return
    x, y, theta = pose
    start = (int(round(x)), int(round(y)))
    tip = (int(round(x + arrow_length * np.cos(theta))), int(round(y + arrow_length * np.sin(theta))))
    cv2.circle(image, start, 8, color, -1)
    cv2.arrowedLine(image, start, tip, color, 4, cv2.LINE_AA, tipLength=0.3)
    draw_label(image, f"{label} ({x:.0f}, {y:.0f}, {np.degrees(theta):.0f} deg)",
               (start[0] + 15, start[1] + 50))


def draw_playground_outline(image, start, end):
    """Red outline from `start` to `end` with labelled corners."""
    height, width = image.shape[:2]
    clamp = lambda p: (min(max(p[0], 0), width - 1), min(max(p[1], 0), height - 1))
    draw_start, draw_end = clamp(start), clamp(end)
    cv2.rectangle(image, draw_start, draw_end, RED, 4)
    cv2.circle(image, draw_start, 10, (0, 200, 0), -1)
    cv2.circle(image, draw_end, 10, (0, 200, 0), -1)
    draw_label(image, f"start {start}", (draw_start[0] + 15, draw_start[1] + 95))
    draw_label(image, f"end {end}", (draw_end[0] - 15, draw_end[1] - 95), align_right=True, below=False)


def draw_obstacles(image, obstacles, labels=True):
    for obstacle in obstacles:
        cv2.polylines(image, [obstacle.corners.astype(np.int32)], True, RED, 4)
        for corner in obstacle.corners:
            cv2.circle(image, tuple(int(v) for v in corner), 8, ORANGE, -1)
    if labels:
        for index, obstacle in enumerate(obstacles):
            cx, cy = (int(round(v)) for v in obstacle.center)
            draw_label(image, f"obstacle {index}", (cx - 60, cy - 15))


def draw_frame(frame):
    """Annotated copy of frame.warped: obstacles, Thymio and goal."""
    image = frame.warped.copy()
    draw_obstacles(image, frame.obstacles, labels=False)
    draw_pose(image, frame.thymio, "thymio", BLUE)
    draw_pose(image, frame.goal, "goal", GREEN)
    # Obstacle labels last so they are never hidden behind the pose labels.
    for index, obstacle in enumerate(frame.obstacles):
        cx, cy = (int(round(v)) for v in obstacle.center)
        draw_label(image, f"obstacle {index}", (cx - 60, cy - 15))
    return image


def show(image, title=None, ax=None, figsize=(14, 7)):
    """Display a BGR image with matplotlib (for notebooks)."""
    import matplotlib.pyplot as plt
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
