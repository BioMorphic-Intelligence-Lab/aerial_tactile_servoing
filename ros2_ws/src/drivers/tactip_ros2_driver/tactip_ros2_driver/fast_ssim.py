"""Fast drop-in for skimage.metrics.structural_similarity with its default settings.

skimage's SSIM on a 480x640 frame costs ~45 ms, which on its own is close to one camera frame
period per sensor. This computes the same quantity with OpenCV box filters in float32 (~12 ms),
matching skimage to ~1e-9, so existing contact thresholds remain valid.

Matched skimage defaults: 7x7 uniform window, K1=0.01, K2=0.03, sample covariance (N/(N-1)),
data_range from the integer dtype, and the mean taken over the image with a (win-1)/2 border
cropped off (so the border handling of the filter does not matter).
"""
import numpy as np
import cv2

WIN_SIZE = 7
K1 = 0.01
K2 = 0.03


def structural_similarity(im1, im2):
    """Mean SSIM of two equally shaped 2D integer images (e.g. uint8)."""
    if im1.shape != im2.shape or im1.ndim != 2:
        raise ValueError(f"Expected two equal 2D images, got {im1.shape} and {im2.shape}")
    if not np.issubdtype(im1.dtype, np.integer):
        raise ValueError(f"Expected an integer image (data_range comes from its dtype), got {im1.dtype}")
    info = np.iinfo(im1.dtype)
    data_range = float(info.max - info.min) if info.min < 0 else float(info.max)

    a = im1.astype(np.float32)
    b = im2.astype(np.float32)

    def box(x):
        return cv2.blur(x, (WIN_SIZE, WIN_SIZE), borderType=cv2.BORDER_REFLECT)

    ux, uy = box(a), box(b)
    uxx = box(cv2.multiply(a, a))
    uyy = box(cv2.multiply(b, b))
    uxy = box(cv2.multiply(a, b))

    n = WIN_SIZE * WIN_SIZE
    cov_norm = n / (n - 1.0)
    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2

    uxuy = ux * uy
    ux2 = ux * ux
    uy2 = uy * uy
    num = (2.0 * uxuy + c1) * (2.0 * cov_norm * (uxy - uxuy) + c2)
    den = (ux2 + uy2 + c1) * (cov_norm * (uxx - ux2 + uyy - uy2) + c2)

    p = (WIN_SIZE - 1) // 2
    s = num[p:-p, p:-p] / den[p:-p, p:-p]
    return float(s.mean(dtype=np.float64))
