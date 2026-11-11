"""
Unit tests for src/noise.py
"""
import numpy as np
from src.noise import (
    estimate_noise,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_gaussian_filter,
    apply_median_filter,
    analyze_noise
)

def test_noise():
    print("Testing Noise Module...")

    # 1. Clean smooth image
    clean_img = np.ones((200, 200), dtype=np.uint8) * 128
    noise_clean = estimate_noise(clean_img)
    print(f"Clean image estimated noise: {noise_clean:.4f}")
    assert noise_clean < 0.5, f"Expected near zero noise, got {noise_clean}"

    # 2. Add Gaussian noise with known sigma=20
    noisy_gauss = add_gaussian_noise(clean_img, sigma=20.0)
    est_gauss = estimate_noise(noisy_gauss)
    print(f"Gaussian noisy image estimated noise (injected sigma=20): {est_gauss:.2f}")
    assert abs(est_gauss - 20.0) < 5.0, f"Expected estimated noise ~20, got {est_gauss}"

    # 3. Add Salt and Pepper noise
    noisy_sp = add_salt_and_pepper_noise(clean_img, amount=0.05)
    est_sp = estimate_noise(noisy_sp)
    print(f"Salt & Pepper noisy image estimated noise: {est_sp:.2f}")
    assert est_sp > 10.0

    # 4. Filter comparison on Salt & Pepper noise:
    # Median filter should restore clean image almost perfectly!
    median_restored = apply_median_filter(noisy_sp, ksize=3)
    gauss_restored = apply_gaussian_filter(noisy_sp, ksize=3)

    diff_median = np.mean(np.abs(median_restored.astype(float) - clean_img.astype(float)))
    diff_gauss = np.mean(np.abs(gauss_restored.astype(float) - clean_img.astype(float)))
    print(f"Mean Abs Error after Median Filter: {diff_median:.2f}")
    print(f"Mean Abs Error after Gaussian Filter: {diff_gauss:.2f}")
    assert diff_median < diff_gauss, "Median filter must outperform Gaussian filter on Salt & Pepper noise"

    # 5. Full pipeline test
    res = analyze_noise(noisy_gauss, noise_threshold=12.0)
    assert res["noise_status"] == "HIGH"
    print("Noise tests passed successfully!")

if __name__ == "__main__":
    test_noise()

