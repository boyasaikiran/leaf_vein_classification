import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.feature import graycomatrix, graycoprops  # ✅ Updated import names

# ------------------------------------------------------------
# 1️⃣ Extract Veins using Distance Transform + Skeletonization
# ------------------------------------------------------------
def extract_veins(image):
    """Convert image to binary, extract veins and skeleton."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding to create binary mask
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)  # Invert to make veins white

    # Distance transform
    dist = distance_transform_edt(binary)

    # Normalize distance transform for visualization
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Skeletonization
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

    return dist_norm, skeleton


# ------------------------------------------------------------
# 2️⃣ Extract Numerical Features from Veins
# ------------------------------------------------------------
def extract_features(image, binary, skeleton):
    """Extract quantitative features from skeletonized leaf veins."""

    features = []

    # --- 1. Basic Vein Stats ---
    vein_area = np.sum(binary > 0)
    total_area = binary.shape[0] * binary.shape[1]
    vein_density = vein_area / total_area if total_area > 0 else 0

    skeleton_length = np.sum(skeleton > 0)
    skeleton_ratio = skeleton_length / total_area if total_area > 0 else 0

    features.extend([vein_density, skeleton_ratio])

    # --- 2. Region-based Shape Features ---
    labeled = label(binary)
    regions = regionprops(labeled)
    if regions:
        largest = max(regions, key=lambda r: r.area)
        features.extend([
            largest.area,
            largest.perimeter,
            largest.eccentricity,
        ])
    else:
        features.extend([0, 0, 0])

    # --- 3. Texture Features (GLCM) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    features.extend([contrast, homogeneity, energy, correlation])

    # --- 4. Color Features ---
    mean_color = cv2.mean(image)[:3]
    features.extend(mean_color)

    # --- 5. Geometric / Curvature Approximation ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    else:
        circularity = 0
    features.append(circularity)

    return np.array(features)
