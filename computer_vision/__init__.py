"""Computer vision for the Thymio playground.

camera        : open a webcam / load images
transform     : ArUco detection and camera -> top-view playground transform
position      : Thymio / goal pose and obstacle detection
visualization : drawing helpers
"""

from .camera import Camera, load_image
from .transform import (CORNER_IDS, DEFAULT_SIZE, compute_transform, detect_markers,
                        find_playground_transform, playground_rectangle, to_playground,
                        warp_playground)
from .position import (ALL_MARKER_IDS, GOAL_ID, THYMIO_ID, Frame, Obstacle, analyse_frame,
                       aruco_mask, detect_obstacles, get_position, marker_pose)
from .visualization import (draw_frame, draw_label, draw_obstacles, draw_playground_outline,
                            draw_pose, show)
