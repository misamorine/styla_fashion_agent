import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ============================================================
# BODY AXIS
# ============================================================

def get_body_axis(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip
):
    """
    Calculate the approximate body center axis using shoulder and hip centers.
    Returns:
        shoulder_center, hip_center
    """
    shoulder_center = (
        int((left_shoulder[0] + right_shoulder[0]) / 2),
        int((left_shoulder[1] + right_shoulder[1]) / 2)
    )

    hip_center = (
        int((left_hip[0] + right_hip[0]) / 2),
        int((left_hip[1] + right_hip[1]) / 2)
    )

    return shoulder_center, hip_center


# ============================================================
# FACING DIRECTION
# ============================================================

def detect_facing_direction(
    nose,
    shoulder_center,
    hip_center
):
    """
    Determine whether the person is facing LEFT or RIGHT.
    Returns: "left" or "right"
    """
    sy = shoulder_center[1]
    sx = shoulder_center[0]

    hy = hip_center[1]
    hx = hip_center[0]

    if hy == sy:
        axis_x = sx
    else:
        t = (nose[1] - sy) / (hy - sy)
        axis_x = sx + t * (hx - sx)

    if nose[0] < axis_x:
        return "left"

    return "right"


# ============================================================
# BODY AXIS X AT A GIVEN Y
# ============================================================

def get_axis_x_at_y(
    y,
    shoulder_center,
    hip_center
):
    """Get body-axis x coordinate at a particular y row."""
    sy = shoulder_center[1]
    sx = shoulder_center[0]

    hy = hip_center[1]
    hx = hip_center[0]

    if hy == sy:
        return sx

    t = (y - sy) / (hy - sy)

    return sx + t * (hx - sx)


# ============================================================
# SEARCH REGIONS
# ============================================================

def find_profile_search_regions(
    shoulder_center,
    hip_center
):
    """
    Create approximate anatomical search regions for bust, belly and butt.
    """
    shoulder_y = shoulder_center[1]
    hip_y = hip_center[1]

    torso_height = hip_y - shoulder_y

    if torso_height <= 0:
        raise ValueError("Invalid pose: hip is above shoulder.")

    regions = {
        "bust": (
            shoulder_y + int(0.18 * torso_height),
            shoulder_y + int(0.38 * torso_height)
        ),
        "belly": (
            shoulder_y + int(0.65 * torso_height),
            shoulder_y + int(0.90 * torso_height)
        ),
        "butt": (
            shoulder_y + int(0.68 * torso_height),
            hip_y + int(0.18 * torso_height)
        )
    }

    return regions


# ============================================================
# BODY CONTOUR
# ============================================================

def get_largest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


# ============================================================
# CONTOUR EDGES AT EACH ROW
# ============================================================

def get_row_edges(mask, y):
    """Get left and right body edges at a row."""
    if y < 0 or y >= mask.shape[0]:
        return None

    white_pixels = np.where(mask[y] == 255)[0]

    if len(white_pixels) == 0:
        return None

    return (
        int(white_pixels[0]),
        int(white_pixels[-1])
    )


# ============================================================
# PROTRUSION PROFILE
# ============================================================

def get_protrusion_profile(
    mask,
    shoulder_center,
    hip_center,
    facing_direction,
    start_y,
    end_y
):
    """Calculate protrusion distance from the body axis."""
    profile = []

    for y in range(start_y, end_y):
        edges = get_row_edges(mask, y)

        if edges is None:
            continue

        left_x, right_x = edges

        axis_x = get_axis_x_at_y(
            y,
            shoulder_center,
            hip_center
        )

        if facing_direction == "right":
            front_distance = right_x - axis_x
            back_distance = axis_x - left_x
        else:
            front_distance = axis_x - left_x
            back_distance = right_x - axis_x

        profile.append({
            "y": y,
            "left": left_x,
            "right": right_x,
            "axis_x": axis_x,
            "front_distance": front_distance,
            "back_distance": back_distance
        })

    return profile


def find_profile_point(
    mask,
    shoulder_center,
    hip_center,
    facing_direction,
    region,
    side
):
    """Detect anatomical protrusion using local contour shape."""
    start_y, end_y = region

    profile = get_protrusion_profile(
        mask,
        shoulder_center,
        hip_center,
        facing_direction,
        start_y,
        end_y
    )

    if len(profile) < 15:
        return None

    if side == "front":
        if facing_direction == "right":
            contour_x = np.array([p["right"] for p in profile], dtype=float)
        else:
            contour_x = np.array([p["left"] for p in profile], dtype=float)
    else:
        if facing_direction == "right":
            contour_x = np.array([p["left"] for p in profile], dtype=float)
        else:
            contour_x = np.array([p["right"] for p in profile], dtype=float)

    smooth_x = gaussian_filter1d(contour_x, sigma=5)

    if side == "front":
        signal = smooth_x if facing_direction == "right" else -smooth_x
    else:
        signal = -smooth_x if facing_direction == "right" else smooth_x

    peaks, properties = find_peaks(signal, prominence=2, distance=10)

    if len(peaks) == 0:
        best_index = int(np.argmax(signal))
    else:
        best_index = int(peaks[np.argmax(signal[peaks])])

    best = profile[best_index]

    if side == "front":
        x = best["right"] if facing_direction == "right" else best["left"]
    else:
        x = best["left"] if facing_direction == "right" else best["right"]

    return {
        "x": int(x),
        "y": int(best["y"]),
        "axis_x": int(best["axis_x"]),
        "distance": float(abs(signal[best_index]))
    }


def get_curved_body_axis(
    mask,
    shoulder_center,
    hip_center
):
    """Estimate a smooth curved body axis between shoulder and hip."""
    shoulder_y = shoulder_center[1]
    hip_y = hip_center[1]

    start_y = min(shoulder_y, hip_y)
    end_y = max(shoulder_y, hip_y)

    height = end_y - start_y
    if height <= 0:
        return []

    axis_points = []

    for y in range(start_y, end_y + 1):
        row = mask[y]
        white_pixels = np.where(row == 255)[0]

        if len(white_pixels) < 2:
            continue

        left = white_pixels[0]
        right = white_pixels[-1]
        center_x = (left + right) / 2.0
        axis_points.append((y, center_x))

    if len(axis_points) < 10:
        return axis_points

    ys = np.array([p[0] for p in axis_points], dtype=float)
    xs = np.array([p[1] for p in axis_points], dtype=float)
    smooth_xs = gaussian_filter1d(xs, sigma=20)

    t = (ys - start_y) / (end_y - start_y)
    straight_xs = shoulder_center[0] * (1 - t) + hip_center[0] * t
    smooth_xs = 0.25 * smooth_xs + 0.75 * straight_xs

    curved_axis = []
    for y, x in zip(ys, smooth_xs):
        curved_axis.append((int(y), int(round(x))))

    return curved_axis
