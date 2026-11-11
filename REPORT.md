# ACADEMIC PROJECT REPORT & VIVA DEFENSE GUIDE
**Project Title:** Automated E-commerce Image Quality and Compliance Inspector  
**Subject:** Digital Image Processing (DIP)  
**Student Profile:** B.Tech Information Technology  
**Industry Benchmark:** Neuropixel.ai | Reference ID: HSR-AIML-056  
**Dataset Source:** Amazon Berkeley Objects (ABO) Dataset  

---

## 1. Abstract
Visual quality and catalog consistency directly influence customer conversion and brand trust on e-commerce platforms. Sellers frequently upload catalog assets suffering from blur, uneven lighting, flat contrast, sensor noise, irregular backdrops, and accidental edge-clipping. 

This project develops an automated image inspection and compliance verification system grounded strictly in classical Digital Image Processing (DIP) techniques complemented by an interpretable Machine Learning decision engine. Sourced from the benchmark **Amazon Berkeley Objects (ABO)** dataset, physical catalog products (furniture, kitchenware, shoes, bags, home decor, electronics) are evaluated across seven discrete DIP stages, extracting an 8-dimensional feature vector quantifying physical image characteristics:
1. Mean pixel intensity (Brightness mu)
2. Intensity standard deviation (Contrast sigma)
3. Statistical variance of the discrete Laplacian operator (Var(nabla^2 I))
4. High-frequency Immerkaer noise proxy (sigma_n)
5. Foreground-to-canvas area ratio
6. Spatial moment centroid alignment score
7. Background uniformity score
8. Bounding box boundary proximity ratio

Two decision frameworks are implemented and quantitatively compared: a deterministic **Rule-Based Baseline** and an **Improved Random Forest Classifier**. On a rigorously isolated held-out test dataset, the Random Forest model achieves **80.0% accuracy** and reduces the False Reject Rate from **60.0% down to 20.0%**, demonstrating why multidimensional statistical learning outperforms rigid heuristic thresholds. The entire pipeline is wrapped in an interactive local Streamlit dashboard.

---

## 2. Theoretical Background & Viva Questions

### Q1: Why convert RGB images to Grayscale using weighted coefficients rather than simple averaging?
**Answer:** The human visual system (photopic vision) relies on retinal cone cells with differing spectral sensitivities: high sensitivity to green wavelengths (555 nm), moderate to red, and low to blue. Simple averaging (R+G+B)/3 distorts perceived luminance. We apply the ITU-R BT.601 standard:
Y = 0.299 * R + 0.587 * G + 0.114 * B
This creates a grayscale representation reflecting genuine perceived brightness.

### Q2: How does the Laplacian Operator detect blur? What is the mathematical intuition?
**Answer:** The Laplacian is a 2nd-order spatial differential operator defined as:
nabla^2 f(x, y) = (d^2 f / dx^2) + (d^2 f / dy^2)
In discrete spatial convolution, it is approximated with the 4-connected kernel:
[ 0,  1,  0 ]
[ 1, -4,  1 ]
[ 0,  1,  0 ]
Edges represent rapid intensity shifts. At an edge, the first derivative reaches an extremum, and the second derivative produces a zero-crossing surrounded by strong positive and negative spikes. A crisp, sharp image generates large response spikes across edges, resulting in high statistical variance Var(nabla^2 I). Blurry images have smoothed, gradual transitions, producing shallow second derivatives and low variance.

### Q3: Why use the Immerkaer (1996) formulation for noise estimation without a ground-truth clean image?
**Answer:** In single-image inspection, reference ground-truth images do not exist. Immerkaer convolved the image with a Laplacian-like mask:
M = [  1, -2,  1 ]
    [ -2,  4, -2 ]
    [  1, -2,  1 ]
Because sum(M) = 0, the mask completely annihilates constant regions, linear intensity gradients, and smooth geometric contours. What survives the convolution is high-frequency zero-mean additive noise. Scaled by sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))), it estimates the true noise standard deviation sigma_n directly from a single degraded input.

### Q4: Why does a Median Filter eliminate Salt-and-Pepper noise while a Gaussian Filter fails?
**Answer:**
- **Gaussian Filter (Linear):** Performs weighted local averaging. When a pixel with intensity 0 or 255 (salt/pepper spike) is averaged with its neighbors, the extreme value is diffused across adjacent pixels, smearing the artifact into an objectionable blurred patch.
- **Median Filter (Non-linear Order-Statistic):** Sorts neighborhood pixel values and selects the middle rank value. Because impulse spikes are statistical outliers residing exclusively at rank 0 or rank k, the median operation discards them completely, preserving edge sharpness and restoring clean surfaces.

### Q5: How does Otsu's Thresholding separate foreground products from studio backgrounds?
**Answer:** Otsu's algorithm calculates the global threshold t* that maximizes between-class variance sigma_B^2(t):
sigma_B^2(t) = w0(t) * w1(t) * [mu0(t) - mu1(t)]^2
where w0(t) and w1(t) are cumulative class probabilities, and mu0(t) and mu1(t) are mean intensities of background and foreground. Maximizing between-class variance is mathematically equivalent to minimizing intra-class variance, ensuring optimal bi-modal separation.

### Q6: What roles do Morphological Opening and Closing play on the segmentation mask?
**Answer:**
- **Opening:** Erosion followed by dilation. It smooths contour boundaries, breaks thin isthmuses, and removes isolated noise specks (pepper noise).
- **Closing:** Dilation followed by erosion. It bridges narrow fractures, closes small contour gaps, and fills interior pinholes within the product mask.

### Q7: How are spatial image moments used to calculate the product Centroid?
**Answer:** For a 2D binary shape I(x, y), the (p, q)-th moment is:
m_{p,q} = sum_{x,y} x^p * y^q * I(x, y)
- m00 equals the total pixel area of the shape.
- m10 and m01 are intensity-weighted coordinate sums along X and Y.
The center of mass (Centroid) is calculated as:
Cx = m10 / m00,  Cy = m01 / m00

---

## 3. Dataset Construction & Zero-Leakage Protocol
To ensure valid scientific evaluation, 100 clean product items were sampled from the **Amazon Berkeley Objects (ABO)** dataset.
- **Filtering:** Selected physical product assets on clean studio white backgrounds (mean border intensity > 215) with no human fashion models.
- **Partitioning Rule:** Sourced base Amazon product IDs were partitioned 80/20 into Train (80) and Test (20) sets prior to generating any defects.
- **Defect Synthesis:** Controlled synthetic variations were injected strictly within partitions:
  - Low Brightness: I * 0.25
  - High Brightness: I + 120
  - Low Contrast: dynamic range compressed to [105, 145]
  - Gaussian Blur: 21 x 21 kernel, sigma = 8
  - Motion Blur: 1D horizontal directional averaging kernel (length 17)
  - Gaussian Noise: sigma = 30
  - Salt-and-Pepper: 6% random pixel saturation
- **Traceability:** Full provenance recorded in dataset/manifest.csv.

---

## 4. Quantitative Results & Critical Analysis

### Performance Comparison Table (Held-Out Test Set)

| Metric | Baseline (Fixed Rules) | Improved (Random Forest) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 65.00% | **80.00%** | ML improves general diagnostic accuracy by 15% |
| **Precision** | 60.00% | **80.00%** | Higher confidence defect detection |
| **Recall** | 90.00% | **80.00%** | High defect catch rate |
| **F1 Score** | 0.7200 | **0.8000** | Superior balanced performance |
| **False Reject Rate (FRR)** | **60.00%** | **20.00%** | **Cuts wrongful seller rejections by 3x** |

### Confusion Matrices:

**Baseline (Fixed Rules):**
- True Accept: 4
- False Reject: 6 (FRR = 6/10 = 60.00%)
- False Accept: 1
- True Reject: 9

**Improved (Random Forest):**
- True Accept: 8
- False Reject: 2 (FRR = 2/10 = 20.00%)
- False Accept: 2
- True Reject: 8

---

## 5. Summary of Deliverables
1. **Core Processing Modules:** Clean, documented implementations in `src/` covering Stages 1 through 8.
2. **Deterministic Baseline:** Rule engine in `src/baseline.py` providing human-readable explanations.
3. **Machine Learning Model:** Serialized Random Forest model in `models/random_forest.joblib`.
4. **Interactive Dashboard:** Complete 6-section Streamlit app in `app.py`.
5. **Validation Test Suite:** Automated test suite in `tests/test_end_to_end.py` verifying all 6 required conditions.
6. **Documentation:** `README.md` and `REPORT.md` ready for viva examination.
