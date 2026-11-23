"""ArUco detection and camera image -> top-view playground transform.

Playground coordinate system (pixels of the top view):
    (0, 0)                      = top-left corner of ArUco 0
    (width, height)             = bottom-right corner of ArUco 2
    x to the right, y downwards.
"""

import cv2
import numpy as np

ARUCO_DICTIONARY = cv2.aruco.DICT_6X6_250
CORNER_IDS = (0, 1, 2, 3)        # top-left, top-right, bottom-right, bottom-left
DEFAULT_SIZE = (1500, 1000)      # (width, height) of the top view in pixels


def _detector_parameters(threshold_constant):
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:  # OpenCV < 4.7
        parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = threshold_constant
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return parameters


def _detect_once(channel, dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        quads, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(channel)
    else:
        quads, ids, _ = cv2.aruco.detectMarkers(channel, dictionary, parameters=parameters)
    if ids is None or len(ids) == 0:
        return {}
    return {int(i): q.reshape(4, 2).astype(np.float32) for q, i in zip(quads, ids.ravel())}


def detect_markers(image, marker_ids=None):
    """Return {marker_id: (4, 2) float32 corners} in camera-image pixels.

    Corners are in ArUco order: the marker's own top-left, top-right, bottom-right,
    bottom-left. The corner markers sit on the paper edge with no white margin,
    so a single detector pass misses some of them; several preprocessed versions
    of the image are tried (closing / median remove the thin paper-edge shadow,
    the red channel makes the beige carpet bright) and the detections are merged.

    marker_ids : iterable of IDs to look for. None = keep every marker found
                 (all passes are run). With IDs, stops as soon as all are found.
    """
    wanted = None if marker_ids is None else set(marker_ids)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    red = image[:, :, 2] if image.ndim == 3 else image
    kernel = np.ones((5, 5), np.uint8)
    passes = [
        lambda: (gray, 7),
        lambda: (cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel), 15),
        lambda: (cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel), 15),
        lambda: (cv2.medianBlur(gray, 5), 15),
        lambda: (red, 15),
    ]

    found = {}
    for make_pass in passes:
        channel, threshold_constant = make_pass()
        for marker_id, quad in _detect_once(channel, dictionary, _detector_parameters(threshold_constant)).items():
            if (wanted is None or marker_id in wanted) and marker_id not in found:
                found[marker_id] = quad
        if wanted is not None and wanted <= found.keys():
            break
    return found


def compute_transform(markers, size=DEFAULT_SIZE, margin=0):
    """Perspective transform (3x3) from camera pixels to playground pixels.

    markers : output of detect_markers(); must contain the four CORNER_IDS.
    The outer corner of each corner marker (farthest from the field centre)
    is mapped to a corner of the top view, so all four markers are fully visible.
    Raises ValueError if a corner marker is missing.
    """
    missing = [i for i in CORNER_IDS if i not in markers]
    if missing:
        raise ValueError(f"Corner markers not detected: {missing} (found {sorted(markers)})")
    width, height = int(size[0]), int(size[1])
    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError("Output size must be larger than twice the margin")

    quads = [markers[i] for i in CORNER_IDS]
    field_center = np.mean([q.mean(axis=0) for q in quads], axis=0)
    outer = np.array([q[np.argmax(np.linalg.norm(q - field_center, axis=1))] for q in quads],
                     dtype=np.float32)
    # Order by position in the image: top-left, top-right, bottom-right, bottom-left.
    sums = outer.sum(axis=1)
    diffs = outer[:, 1] - outer[:, 0]
    source = np.array([outer[np.argmin(sums)], outer[np.argmin(diffs)],
                       outer[np.argmax(sums)], outer[np.argmax(diffs)]], dtype=np.float32)
    m = int(margin)
    destination = np.array([[m, m], [width - m, m], [width - m, height - m], [m, height - m]],
                           dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def find_playground_transform(image, size=DEFAULT_SIZE, margin=0):
    """Detect the corner markers in `image` and return the playground transform."""
    return compute_transform(detect_markers(image, CORNER_IDS), size, margin)


def warp_playground(image, transform, size=DEFAULT_SIZE):
    """Top view of the playground, `size` = (width, height)."""
    return cv2.warpPerspective(image, transform, (int(size[0]), int(size[1])))


def to_playground(points, transform):
    """Map camera-image points (N, 2) to playground coordinates (N, 2)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, transform).reshape(-1, 2)


def playground_rectangle(markers, transform):
    """(start, end) of the playground: top-left of ArUco 0 and bottom-right of ArUco 2."""
    if 0 not in markers or 2 not in markers:
        raise ValueError(f"ArUco 0 and 2 must be detected (found {sorted(markers)})")
    corners_0 = to_playground(markers[0], transform)
    corners_2 = to_playground(markers[2], transform)
    start = corners_0[np.argmin(corners_0.sum(axis=1))]
    end = corners_2[np.argmax(corners_2.sum(axis=1))]
    return tuple(int(round(v)) for v in start), tuple(int(round(v)) for v in end)
