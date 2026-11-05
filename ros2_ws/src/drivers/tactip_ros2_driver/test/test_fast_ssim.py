import numpy as np
import cv2
import pytest
from skimage.metrics import structural_similarity as skimage_ssim

from tactip_ros2_driver.fast_ssim import structural_similarity as fast_ssim


def _pins(shift=(0, 0), warp=0.0, seed=0):
    """Synthetic TacTip-like frame: bright pins on a dark background, thresholded to 0/255."""
    rng = np.random.default_rng(seed)
    img = np.full((480, 640), 40, np.uint8)
    for y in range(40, 460, 22):
        for x in range(120, 540, 22):
            dx = shift[0] + warp * np.sin(y / 40.0) + rng.normal(0, 0.5)
            dy = shift[1] + warp * np.cos(x / 40.0) + rng.normal(0, 0.5)
            cv2.circle(img, (int(x + dx), int(y + dy)), 5, 230, -1)
    return np.where(cv2.GaussianBlur(img, (5, 5), 0) > 128, 255, 0).astype(np.uint8)


@pytest.mark.parametrize("other", [
    _pins(seed=1),
    _pins(shift=(3, 2), seed=2),
    _pins(warp=6, seed=3),
])
def test_matches_skimage_on_tactip_like_frames(other):
    ref = _pins()
    assert fast_ssim(ref, other) == pytest.approx(skimage_ssim(ref, other), abs=1e-6)


def test_matches_skimage_on_random_frames():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.integers(0, 256, (120, 160), dtype=np.uint8)
        b = cv2.GaussianBlur(a, (5, 5), 0)
        assert fast_ssim(a, b) == pytest.approx(skimage_ssim(a, b), abs=1e-6)


def test_identical_images_score_one():
    a = _pins()
    assert fast_ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_rejects_float_and_mismatched_inputs():
    a = _pins()
    with pytest.raises(ValueError):
        fast_ssim(a.astype(np.float32), a.astype(np.float32))
    with pytest.raises(ValueError):
        fast_ssim(a, a[:-1])
