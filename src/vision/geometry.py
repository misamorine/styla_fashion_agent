import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def get_largest_contour(mask):

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    return largest

def get_width_profile(mask):
    height, width = mask.shape
    width_profile = []
    for y in range(height):
        row = mask[y]
        white_pixels = np.where(row == 255)[0]
        if len(white_pixels) == 0:
            width_profile.append(0)
            continue
        
        left = white_pixels[0]
        right = white_pixels[-1]
        body_width = right - left
        width_profile.append(body_width)

    return width_profile

def smooth_width_profile(width_profile):
    """
    Smooth the width profile using a Gaussian filter.
    """

    smooth_profile = gaussian_filter1d(
        width_profile,
        sigma=4
    )

    return smooth_profile
def find_search_regions(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
    left_knee,
    right_knee
):
    """
    Returns approximate search windows for
    bust, waist and hip.
    """

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
    hip_y = (left_hip[1] + right_hip[1]) // 2
    knee_y = (left_knee[1] + right_knee[1]) // 2

    torso_height = hip_y - shoulder_y

    search_rows = {

        # Around chest
        "bust": (
            shoulder_y + int(0.22 * torso_height),
            shoulder_y + int(0.42 * torso_height)
        ),

        # Middle torso
        "waist": (
            shoulder_y + int(0.35 * torso_height),
            shoulder_y + int(0.75 * torso_height)
        ),

        # Around hips
        "hip": (
            hip_y,
            hip_y + int(0.30 * (knee_y - hip_y))
        )
    }

    return search_rows

def detect_measurement_rows(
    smooth_profile,
    search_rows
):
    """
    Detect bust, waist and hip rows
    using peak detection.
    """

    results = {}

    for region, (start, end) in search_rows.items():

        region_profile = smooth_profile[start:end]

        if region == "waist":

            peaks, _ = find_peaks(-region_profile)

        else:

            peaks, _ = find_peaks(region_profile)

        if len(peaks) == 0:

            if region == "waist":
                row = np.argmin(region_profile)
            else:
                row = np.argmax(region_profile)

        else:

            if region == "waist":

                best_peak = peaks[
                    np.argmax((-region_profile)[peaks])
                ]

            else:

                best_peak = peaks[
                    np.argmax(region_profile[peaks])
                ]

            row = start + best_peak

        results[region] = row

    return results

def estimate_bust_row(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
    smooth_profile
):
    """
    Detect bust row using anatomy + width profile.

    Strategy:
    1. Estimate anatomical bust region.
    2. Search for all width peaks.
    3. Ignore peaks too close to shoulders.
    4. Choose the widest remaining peak.
    5. If no valid peak exists, use anatomical estimate.
    """

    shoulder_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
    hip_y = int((left_hip[1] + right_hip[1]) / 2)

    torso_height = hip_y - shoulder_y

    # Anatomical bust search region
    start = shoulder_y + int(0.22 * torso_height)
    end   = shoulder_y + int(0.42 * torso_height)

    region = smooth_profile[start:end]

    # Find all local maxima
    peaks, _ = find_peaks(region)

    if len(peaks) == 0:
        # Fallback
        return start + np.argmax(region)

    # Ignore upper shoulder/armpit area
    ignore_limit = int(len(region) * 0.40)

    valid_peaks = []

    for peak in peaks:

        if peak < ignore_limit:
            continue

        valid_peaks.append(peak)

    # If everything got ignored
    if len(valid_peaks) == 0:
        valid_peaks = peaks

    # Pick the widest remaining peak
    best_peak = valid_peaks[0]

    for peak in valid_peaks:
        if region[peak] > region[best_peak]:
            best_peak = peak

    bust_row = start + best_peak
    print("Search Region:", start, end) 
    print("All Peaks:", peaks)
    print("Valid Peaks:", valid_peaks)
    for p in valid_peaks:
        print(
        "Peak:",
        start + p,
        "Width:",
        region[p]
        )
    print("Selected Bust:", bust_row)

    return bust_row


def get_body_edges(mask, row):
    """
    Returns left and right body edges
    at a given row.
    """

    white_pixels = np.where(mask[row] == 255)[0]

    if len(white_pixels) == 0:
        return None

    left = white_pixels[0]
    right = white_pixels[-1]

    return left, right

def get_body_width(mask, row):
    """
    Returns body width in pixels
    at a specific row.
    """

    edges = get_body_edges(mask, row)

    if edges is None:
        return 0

    left, right = edges

    return right - left

def get_torso_width(
    mask,
    row,
    left_shoulder,
    right_shoulder
):
    """
    Measure torso width by expanding from body center.
    (Experimental)
    """

    center_x = int(
        (left_shoulder[0] + right_shoulder[0]) / 2
    )

    mask_row = mask[row]

    # Left
    left = center_x
    while left > 0 and mask_row[left] == 255:
        left -= 1

    # Right
    right = center_x
    while right < len(mask_row) - 1 and mask_row[right] == 255:
        right += 1

    return right - left
