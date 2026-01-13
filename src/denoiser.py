import numpy as np
import cv2
from .estimator import RobustNoiseEstimator
from .features import StructureFeatureExtractor

class ALSD:
    def __init__(self):
        self.estimator = RobustNoiseEstimator()
        self.extractor = StructureFeatureExtractor()
        
    def denoise(self, noisy_img):
        # 1. Estimate global noise level
        sigma_n = self.estimator.estimate_sigma(noisy_img)
        print(f"Estimated Noise Sigma: {sigma_n:.2f}")
        
        # 2. Extract features
        features = self.extractor.extract(noisy_img)
        local_var = features['variance']
        
        # 3. Calculate Wiener Weight (Local SNR)
        # Signal Var = Observed Var - Noise Var
        var_n = (sigma_n / 255.0) ** 2
        var_signal = np.maximum(local_var - var_n, 0)
        
        # w = Signal / (Signal + Noise)
        weight_map = var_signal / (var_signal + var_n * 2.0 + 1e-8)
        
        # 4. Filter generation
        img_float = noisy_img.astype(np.float32) / 255.0
        
        # Base filter (Gaussian)
        base = cv2.GaussianBlur(img_float, (0,0), 2.0)
        # Structure filter (Bilateral)
        struct = cv2.bilateralFilter(img_float, 9, 0.3, 2.0)
        
        # Gradient-guided mixing
        grad_w = np.clip(features['gradient'] * 5.0, 0, 1)
        fused_filter = grad_w * struct + (1 - grad_w) * base
        
        # 5. Final Wiener Fusion
        result = weight_map * img_float + (1 - weight_map) * fused_filter
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
