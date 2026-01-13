import numpy as np
import cv2

class RobustNoiseEstimator:
    """Implements robust statistical estimation using MAD."""
    
    @staticmethod
    def estimate_sigma(image: np.ndarray) -> float:
        img_float = image.astype(np.float32)
        # Laplacian-like kernel to suppress low-frequency signal
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        residuals = cv2.filter2D(img_float, -1, kernel)
        
        # MAD estimation
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        
        # Calibration factor for Gaussian distribution
        sigma = mad / 0.6745
        
        # Empirical correction for kernel gain
        sigma_corrected = sigma / np.sqrt(20.0) * 1.5
        return float(sigma_corrected)
