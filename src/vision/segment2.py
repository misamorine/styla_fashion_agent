from fashn_human_parser import FashnHumanParser
import matplotlib.pyplot as plt
import numpy as np
from .body_masks import create_mask, LABELS,create_measurement_mask
import cv2
from . import geometry
from . import pose
from . import classifier

def detect_body_shape(image_path):
    parser = FashnHumanParser()
# Initialize parser (automatically uses GPU if available)
    # image_path = r"imgs\front-full-2.jpeg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Image not found")
    # Run prediction on an image
    segmentation = parser.predict(image_path)
    
    measurement_mask = create_measurement_mask(segmentation)
    #debug
    # plt.figure(figsize=(5,8))
    # plt.imshow(measurement_mask, cmap="gray")
    # plt.title("Measurement Mask")
    # plt.axis("off")
    # plt.show()
    # cv2.imwrite("out/measurement_mask.png", measurement_mask)
    REQUIRED_MASKS = [
        "TOP",
        "DRESS",
        "TORSO",
        "PANTS",
        "SKIRT",
        "LEGS"
    ]

    # masks = {}

    # for label in REQUIRED_MASKS:
    #     masks[label] = create_mask(
    #         segmentation,
    #         LABELS[label]
    #     )

    # for label, mask in masks.items():
    #     cv2.imwrite(f"out/{label.lower()}.png", mask)

    # for label, mask in masks.items():
    #     plt.figure(figsize=(4,6))
    #     plt.imshow(mask, cmap="gray")
    #     plt.title(label)
    #     plt.axis("off")
    #     plt.show()

    # segmentation is a numpy array of shape (H, W) with values 0-17
    # representing the 18 semantic classes
    # print(segmentation.shape)
    # print(segmentation.dtype)
    # print("Unique classes:", np.unique(segmentation))



    # plt.figure(figsize=(6,10))
    # plt.imshow(segmentation)
    # plt.colorbar()
    # plt.title("Segmentation")
    # plt.show()

    # plt.imsave("out/parsing.png", segmentation)
    # Get body contour
    contour = geometry.get_largest_contour(measurement_mask)

    #print(contour)
    # Convert grayscale mask to BGR so we can draw colored contour
    contour_image = cv2.cvtColor(
        measurement_mask,
        cv2.COLOR_GRAY2BGR
    )

    # Draw contour
    cv2.drawContours(
        contour_image,
        [contour],   # Notice the square brackets
        -1,
        (0,255,0),
        2
    )

    # plt.figure(figsize=(5,8))
    # plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    # plt.title("Body Contour")
    # plt.axis("off")
    # plt.show()
    #debug

    width_profile = geometry.get_width_profile(measurement_mask)
    '''old'''
    # print(type(width_profile))
    # print("Number of rows:", len(width_profile))
    # print("First 20 widths:", width_profile[:20])
    # plt.figure(figsize=(6, 10))

    # plt.plot(width_profile)

    # plt.title("Width Profile")
    # plt.xlabel("Image Height (y)")
    # plt.ylabel("Body Width (pixels)")

    # plt.grid(True)

    # plt.show()
    '''new'''
    smooth_profile = geometry.smooth_width_profile(
        width_profile
    )
    #debug
    # plt.figure(figsize=(10,6))

    # plt.plot(width_profile, alpha=0.3, label="Original")

    # plt.plot(smooth_profile,
    #         linewidth=3,
    #         label="Smoothed")

    # plt.legend()

    # plt.grid(True)

    # plt.show()

    mp_image, pose_result = pose.detect_pose(image_path)

    landmarks = pose_result.pose_landmarks[0]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    left_knee = landmarks[25]
    right_knee = landmarks[26]




    image_height, image_width = measurement_mask.shape
    left_shoulder = (
        int(left_shoulder.x * image_width),
        int(left_shoulder.y * image_height)
    )

    right_shoulder = (
        int(right_shoulder.x * image_width),
        int(right_shoulder.y * image_height)
    )

    left_hip = (
        int(left_hip.x * image_width),
        int(left_hip.y * image_height)
    )

    right_hip = (
        int(right_hip.x * image_width),
        int(right_hip.y * image_height)
    )

    left_knee = (
        int(left_knee.x * image_width),
        int(left_knee.y * image_height)
    )

    right_knee = (
        int(right_knee.x * image_width),
        int(right_knee.y * image_height)
    )

    search_rows = geometry.find_search_regions(
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        left_knee,
        right_knee
    )

    # print(search_rows)

    measurement_rows = geometry.detect_measurement_rows(
        smooth_profile,
        search_rows
    )

    bust_row = geometry.estimate_bust_row(
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        smooth_profile
    )

    measurement_rows["bust"] = bust_row

    # print(measurement_rows)

    bust_width = geometry.get_body_width(
        measurement_mask,
        measurement_rows["bust"]
    )

    # bust_width = geometry.get_torso_width(
    #     measurement_mask,
    #     measurement_rows["bust"],
    #     left_shoulder,
    #     right_shoulder
    # )

    waist_width = geometry.get_body_width(
        measurement_mask,
        measurement_rows["waist"]
    )

    hip_width = geometry.get_body_width(
        measurement_mask,
        measurement_rows["hip"]
    )

    # print()

    # print("Bust Width :", bust_width)
    # print("Waist Width:", waist_width)
    # print("Hip Width  :", hip_width)


    landmark_image = image.copy()
    important_points = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        left_knee,
        right_knee
    ]

    for point in important_points:

        cv2.circle(
            landmark_image,
            point,
            8,
            (0,0,255),
            -1
        )
    # plt.figure(figsize=(6,10))
    # plt.imshow(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))
    # plt.title("Pose Landmarks")
    # plt.axis("off")
    # plt.show()

    preview = image.copy()

    # Waist search region
    cv2.rectangle(
        preview,
        (0, search_rows["waist"][0]),
        (image_width, search_rows["waist"][1]),
        (255, 255, 0),
        2
    )

    # Hip search region
    cv2.rectangle(
        preview,
        (0, search_rows["hip"][0]),
        (image_width, search_rows["hip"][1]),
        (0, 255, 255),
        2
    )

    # Bust search region
    cv2.rectangle(
        preview,
        (0, search_rows["bust"][0]),
        (image_width, search_rows["bust"][1]),
        (255, 0, 255),
        2
    )

    # plt.figure(figsize=(6,10))
    # plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    # plt.title("Search Regions")
    # plt.axis("off")
    # plt.show()
    #debug

    preview = image.copy()

    cv2.line(
        preview,
        (0, measurement_rows["bust"]),
        (image_width, measurement_rows["bust"]),
        (255, 0, 0),
        2
    )

    cv2.line(
        preview,
        (0, measurement_rows["waist"]),
        (image_width, measurement_rows["waist"]),
        (0, 255, 0),
        2
    )

    cv2.line(
        preview,
        (0, measurement_rows["hip"]),
        (image_width, measurement_rows["hip"]),
        (0, 0, 255),
        2
    )

    center_x = int(
        (left_shoulder[0] + right_shoulder[0]) / 2
    )

    cv2.circle(
        preview,
        (center_x, measurement_rows["bust"]),
        6,
        (255,255,0),
        -1
    )

    # plt.figure(figsize=(6,10))
    # plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Measurement Rows")
    # plt.axis("off")
    # plt.show()
    #debug

    # for i, landmark in enumerate(landmarks):

    #     x = int(landmark.x * image_width)
    #     y = int(landmark.y * image_height)

    #     print(i, x, y)


    shape = classifier.classify_front_shape(
        bust_width,
        waist_width,
        hip_width
    )

    # print()
    # print("----------------")
    # print("BODY SHAPE:", shape)
    # print("----------------")
    return shape