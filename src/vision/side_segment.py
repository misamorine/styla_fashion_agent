import cv2
import numpy as np
from . import pose
from . import profile_geometry
from . import profile_classifier
from .body_masks import create_profile_measurement_mask, create_measurement_mask

def detect_side_shape(image_path: str) -> str:
    """
    Detect side-view body shape from a side-profile full-body photograph.

    Returns:
        Side shape code: 'I', 'P', 'b', 'B', 'S', 'd', 'db', or 'dB'.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_height, image_width = image.shape[:2]

    # Try FashnHumanParser if available, otherwise create adaptive mask from pose
    try:
        from fashn_human_parser import FashnHumanParser
        parser = FashnHumanParser()
        segmentation = parser.predict(image_path)
        profile_mask = create_profile_measurement_mask(segmentation)
    except Exception:
        # Fallback to thresholding image body area
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, profile_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Pose detection
    mp_image, pose_result = pose.detect_pose(image_path)
    if not pose_result.pose_landmarks:
        # Default to I if pose cannot be clearly landmarked
        return "I"

    landmarks = pose_result.pose_landmarks[0]

    def get_point(idx):
        lm = landmarks[idx]
        return (int(lm.x * image_width), int(lm.y * image_height))

    nose = get_point(0)
    left_shoulder = get_point(11)
    right_shoulder = get_point(12)
    left_hip = get_point(23)
    right_hip = get_point(24)

    shoulder_center, hip_center = profile_geometry.get_body_axis(
        left_shoulder, right_shoulder, left_hip, right_hip
    )

    facing_direction = profile_geometry.detect_facing_direction(
        nose, shoulder_center, hip_center
    )

    search_regions = profile_geometry.find_profile_search_regions(
        shoulder_center, hip_center
    )

    bust_point = profile_geometry.find_profile_point(
        profile_mask, shoulder_center, hip_center, facing_direction, search_regions["bust"], "front"
    )

    belly_point = profile_geometry.find_profile_point(
        profile_mask, shoulder_center, hip_center, facing_direction, search_regions["belly"], "front"
    )

    butt_point = profile_geometry.find_profile_point(
        profile_mask, shoulder_center, hip_center, facing_direction, search_regions["butt"], "back"
    )

    torso_height = abs(hip_center[1] - shoulder_center[1])

    if bust_point is not None and belly_point is not None and butt_point is not None:
        side_shape = profile_classifier.classify_side_shape(
            bust_point["distance"],
            belly_point["distance"],
            butt_point["distance"],
            torso_height
        )
        return side_shape

    return "I"
