import os
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

class BrainPreprocessor:
    """
    Fetal brain imaging preprocessor for MRI and ultrasound scans.
    
    Provides:
    1. Multi-format ingestion (PNG, JPG, BMP, TIFF, NIfTI .nii/.nii.gz if available)
    2. Grayscale conversion & intensity percentile clipping (contrast adjustment)
    3. Fetal cranium / brain ROI localization via adaptive thresholding and morphological cleanup
    4. Canonical spatial standardization (target_size: 128x128)
    5. Quality metric extraction (SNR, foreground ratio, mean intensity)
    """

    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def load_image(self, file_path):
        """Loads brain scan from standard or neuroimaging format."""
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Brain image not found: {file_path}")

        # Check for NIfTI format
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            try:
                import nibabel as nib
                nii = nib.load(file_path)
                data = nii.get_fdata()
                if data.ndim == 3:
                    # Select middle axial slice
                    slice_idx = data.shape[2] // 2
                    img_array = data[:, :, slice_idx]
                elif data.ndim == 2:
                    img_array = data
                else:
                    img_array = data[..., 0]
                # Normalize to 0-255
                img_array = (img_array - np.min(img_array)) / (np.ptp(img_array) + 1e-8) * 255.0
                return img_array.astype(np.uint8)
            except ImportError:
                # If nibabel not installed, raise informative error
                raise ImportError("nibabel is required to read .nii/.nii.gz files. Please install nibabel.")
            except Exception as e:
                raise ValueError(f"Error reading NIfTI brain image: {e}")

        # Standard formats (PNG, JPG, TIFF, BMP)
        try:
            with Image.open(file_path) as img:
                img_gray = img.convert('L')
                return np.array(img_gray, dtype=np.uint8)
        except Exception as e:
            raise ValueError(f"Failed to read image file: {e}")

    def normalize_intensity(self, img_array):
        """
        Applies percentile-based intensity normalization (1st - 99th percentile)
        to suppress acquisition noise and standardize contrast.
        """
        img_float = img_array.astype(np.float32)
        p_low, p_high = np.percentile(img_float, (1, 99))
        
        if p_high > p_low:
            clipped = np.clip(img_float, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low)
        else:
            normalized = img_float / (np.max(img_float) + 1e-8)
            
        return normalized

    def isolate_brain_roi(self, img_normalized):
        """
        Identifies fetal cranial region using Otsu-based thresholding
        and morphological opening/closing to suppress extraneous maternal tissue.
        """
        # Threshold at 15% intensity to separate background
        thresh = max(0.12, np.mean(img_normalized) * 0.5)
        binary_mask = img_normalized > thresh

        # Morphological clean up
        struct = ndimage.generate_binary_structure(2, 2)
        cleaned_mask = ndimage.binary_opening(binary_mask, structure=struct, iterations=1)
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=struct, iterations=2)

        # Find largest connected component (fetal skull / brain)
        labeled, num_features = ndimage.label(cleaned_mask)
        if num_features > 0:
            sizes = ndimage.sum(cleaned_mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            brain_mask = (labeled == largest_label)
            # Fill holes in the cranium mask
            brain_mask = ndimage.binary_fill_holes(brain_mask)
            roi_img = img_normalized * brain_mask
        else:
            brain_mask = np.ones_like(img_normalized, dtype=bool)
            roi_img = img_normalized

        return roi_img, brain_mask

    def resize_canonical(self, img_array):
        """Resizes preprocessed float image to target grid (e.g. 128x128)."""
        # Convert to PIL Image for high-quality antialiased bicubic resizing
        uint8_repr = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(uint8_repr)
        resample_mode = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        resized = pil_img.resize(self.target_size, resample=resample_mode)
        return np.array(resized, dtype=np.float32) / 255.0

    def compute_quality_metrics(self, raw_img, preprocessed_img, mask):
        """Calculates image quality metrics for clinical validation reporting."""
        mean_intensity = float(np.mean(preprocessed_img))
        std_intensity = float(np.std(preprocessed_img))
        snr = round(mean_intensity / (std_intensity + 1e-6), 2)
        fg_ratio = round(float(np.mean(mask)), 3)

        return {
            "mean_intensity": round(mean_intensity, 4),
            "std_intensity": round(std_intensity, 4),
            "estimated_snr": snr,
            "cranial_coverage_ratio": fg_ratio,
            "target_resolution": f"{self.target_size[0]}x{self.target_size[1]}",
        }

    def preprocess(self, raw_img_array):
        """
        Executes end-to-end brain scan preprocessing:
        Raw Image -> Intensity Normalization -> ROI Masking -> Canonical Resize
        """
        norm = self.normalize_intensity(raw_img_array)
        roi, mask = self.isolate_brain_roi(norm)
        canonical = self.resize_canonical(roi)
        metrics = self.compute_quality_metrics(raw_img_array, canonical, mask)
        
        return canonical, metrics
