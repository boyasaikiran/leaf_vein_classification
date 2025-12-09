import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Image not readable: {img_path}")

    # Handle all channel cases safely
    if len(img.shape) == 2:
        # grayscale → BGR
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # RGBA → BGR
        img_color = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 3:
        # already BGR
        img_color = img
    else:
        raise ValueError(f"Unsupported channel number: {img.shape}")

    # Resize (optional)
    img_resized = cv2.resize(img_color, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert for skeletonization
    binary = cv2.bitwise_not(binary)

    # Skeletonize
    skeleton = np.zeros_like(binary)
    temp = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
        temp2 = cv2.subtract(temp, open_img)
        eroded = cv2.erode(temp, element)
        skeleton = cv2.bitwise_or(skeleton, temp2)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return img_resized, binary, skeleton
