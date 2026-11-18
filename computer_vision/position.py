"""Positions on the playground: Thymio / goal pose and black obstacles."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .transform import (CORNER_IDS, DEFAULT_SIZE, compute_transform, detect_markers,
                        to_playground, warp_playground)

THYMIO_ID = 4
GOAL_ID = 5
ALL_MARKER_IDS = (*CORNER_IDS, THYMIO_ID, GOAL_ID)


@dataclass
class Obstacle:
    corners: np.ndarray      # (N, 2) int polygon in playground pixels (usually 4 corners)
    center: tuple            # (x, y) centroid in playground pixels
    area: float              # pixels^2


@dataclass
class Frame:
    warped: np.ndarray                   # top view (BGR)
    transform: np.ndarray                # 3x3 camera -> playground
    thymio: tuple = None                 # (x, y, theta) or None if not visible
    goal: tuple = None                   # (x, y, theta) or None if not visible
    obstacles: list = field(default_factory=list)
    markers: dict = field(default_factory=dict)   # raw detections, camera pixels


def marker_pose(markers, transform, marker_id, heading_offset=0.0):
    """(x, y, theta) of a detected marker in playground pixels, or None if it is missing.

    x, y  : marker centre. theta: radians in (-pi, pi], from +x towards +y
    (clockwise on screen, because y points down); 0 = marker top edge pointing
    right. heading_offset rotates theta if the robot's front is not the marker's top.
    """
    if marker_id not in markers:
        return None
    corners = to_playground(markers[marker_id], transform)
    x, y = corners.mean(axis=0)
    forward = (corners[0] + corners[1]) / 2 - (corners[3] + corners[2]) / 2
    theta = np.arctan2(forward[1], forward[0]) + heading_offset
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return float(x), float(y), float(theta)


def get_position(camera_image, transform, marker_id=THYMIO_ID, heading_offset=0.0):
    """Detect one marker in a camera image and return its (x, y, theta), or None."""
    return marker_pose(detect_markers(camera_image, (marker_id,)), transform, marker_id, heading_offset)


def aruco_mask(markers, transform, size=DEFAULT_SIZE, grow=1.35):
    """Top-view mask (255 = ignore) covering every marker in `markers`, scaled by `grow`."""
    width, height = int(size[0]), int(size[1])
    mask = np.zeros((height, width), np.uint8)
    for quad in markers.values():
        projected = to_playground(quad, transform)
        center = projected.mean(axis=0)
        projected = center + (projected - center) * grow
        cv2.fillConvexPoly(mask, np.round(projected).astype(np.int32), 255)
    return mask


def detect_obstacles(warped_image, ignore_mask=None, min_area=2000):
    """Black obstacles in the top view, sorted top to bottom.

    Dark pixels are found with Otsu's threshold; pixels under `ignore_mask`
    (ArUco markers) are ignored; regions smaller than `min_area` are discarded.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ignore_mask is not None:
        dark[ignore_mask > 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    obstacles = []
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        hull = cv2.convexHull(contour)
        corners = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        moments = cv2.moments(contour)
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        obstacles.append(Obstacle(corners=corners, center=center, area=area))
    obstacles.sort(key=lambda o: (o.center[1], o.center[0]))
    return obstacles


def analyse_frame(camera_image, size=DEFAULT_SIZE, previous_transform=None,
                  heading_offset=0.0, find_obstacles=True):
    """Full pipeline on one camera image -> Frame.

    Markers are detected once and reused for the transform, the poses and the mask.
    If a corner marker is hidden (e.g. by a hand) and `previous_transform` is given,
    that transform is reused instead of failing; otherwise ValueError is raised.
    """
    markers = detect_markers(camera_image, ALL_MARKER_IDS)
    try:
        transform = compute_transform(markers, size)
    except ValueError:
        if previous_transform is None:
            raise
        transform = previous_transform

    warped = warp_playground(camera_image, transform, size)
    obstacles = detect_obstacles(warped, aruco_mask(markers, transform, size)) if find_obstacles else []
    return Frame(
        warped=warped,
        transform=transform,
        thymio=marker_pose(markers, transform, THYMIO_ID, heading_offset),
        goal=marker_pose(markers, transform, GOAL_ID),
        obstacles=obstacles,
        markers=markers,
    )
