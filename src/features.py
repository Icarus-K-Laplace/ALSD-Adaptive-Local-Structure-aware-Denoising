import numpy as np
import cv2
from scipy.ndimage import uniform_filter

class StructureFeatureExtractor:
    """Extracts local statistical and structural features."""
    
    def extract(self, image: np.ndarray):
        img_float = image.astype(np.float32) / 255.0
        
        # Fast local variance calculation
        mean = uniform_filter(img_float, size=5)
        mean_sq = uniform_filter(img_float**2, size=5)
        variance = np.maximum(mean_sq - mean**2, 0)
        
        # Gradient magnitude
        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1)
        gradient = np.sqrt(gx**2 + gy**2)
        
        return {'variance': variance, 'gradient': gradient}
