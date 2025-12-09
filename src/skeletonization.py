import cv2
import numpy as np
from skimage.morphology import skeletonize

def get_skeleton(img_gray):
    # Threshold to segment the leaf
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Distance Transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Binary vein mask
    _, vein_mask = cv2.threshold(dist_norm, 0.2, 1.0, cv2.THRESH_BINARY)
    
    # Skeletonize
    skeleton = skeletonize(vein_mask.astype(bool))
    
    
    return (skeleton * 255).astype('uint8')
