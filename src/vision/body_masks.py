import cv2
import numpy as np

LABELS = {
    "BACKGROUND": 0,
    "FACE": 1,
    "HAIR": 2,
    "TOP": 3,
    "DRESS": 4,
    "SKIRT": 5,
    "PANTS": 6,
    "BELT": 7,
    "BAG": 8,
    "HAT": 9,
    "SCARF": 10,
    "GLASSES": 11,
    "ARMS": 12,
    "HANDS": 13,
    "LEGS": 14,
    "FEET": 15,
    "TORSO": 16,
    "JEWELRY": 17
}

def create_mask(segmentation, class_id):
    """
    Converts one semantic class into a binary mask.
    """
    mask = (segmentation == class_id).astype(np.uint8)
    mask *= 255
    return mask

 
def create_measurement_mask(segmentation):
    required_classes = [
        LABELS["TOP"],
        LABELS["DRESS"],
        LABELS["TORSO"],
        LABELS["PANTS"],
        LABELS["SKIRT"],
        LABELS["LEGS"]
    ]
    mask = np.isin(segmentation, required_classes)
    mask = mask.astype(np.uint8)
    mask = mask * 255
    return mask


def create_profile_measurement_mask(segmentation):
    required_classes = [
        LABELS["TOP"],
        LABELS["DRESS"],
        LABELS["TORSO"],
        LABELS["PANTS"],
        LABELS["SKIRT"],
        LABELS["LEGS"]
    ]
    mask = np.isin(segmentation, required_classes)
    mask = mask.astype(np.uint8)
    mask = mask * 255
    return mask

