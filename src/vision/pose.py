import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/"
    "pose_landmarker/"
    "pose_landmarker_heavy/"
    "float16/1/"
    "pose_landmarker_heavy.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Pose model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Download complete!")
# ----------------------------------
# Create Pose Landmarker
# ----------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(options)


# ----------------------------------
# Detect Pose
# ----------------------------------

def detect_pose(image_path):
    """
    Detect pose landmarks from an image.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    image : mp.Image
        MediaPipe image object.

    detection_result : PoseLandmarkerResult
        Raw MediaPipe detection result.
    """

    image = mp.Image.create_from_file(image_path)

    detection_result = detector.detect(image)

    return image, detection_result