# Automated E-commerce Image Quality and Compliance Inspector
**Comprehensive Academic Project Guide, Implementation Mapping & Viva Defense Blueprint**

* **Course:** Digital Image Processing (DIP) & Applied Machine Learning  
* **Student Level:** B.Tech Information Technology  
* **Dataset:** Official Amazon Berkeley Objects (ABO) Dataset  
* **Architecture:** Classical DIP Feature Engineering + Scikit-Learn Classifier + Streamlit UI  

---

## 🧭 Table of Contents
1. [Why Did Training Take Seconds Instead of Hours? (Clearing the Biggest Doubt)](#1-why-did-training-take-seconds-instead-of-hours)
2. [Master Requirement-to-Code Mapping Table](#2-master-requirement-to-code-mapping-table)
3. [End-to-End Pipeline Walkthrough (Step-by-Step for Your Professor)](#3-end-to-end-pipeline-walkthrough)
4. [File & Module Architecture Guide](#4-file--module-architecture-guide)
5. [DIP Mathematical Formulas Implemented](#5-dip-mathematical-formulas-implemented)
6. [Quantitative Benchmark: Baseline vs. Random Forest](#6-quantitative-benchmark-baseline-vs-random-forest)
7. [How to Present & Demonstrate This to Your Professor (Viva Guide)](#7-how-to-present--demonstrate-this-to-your-professor)
8. [Verification & How to Run Everything](#8-verification--how-to-run-everything)

---

## 1. Why Did Training Take Seconds Instead of Hours?

> [!IMPORTANT]
> **The Crucial Distinction: Classical DIP Feature Engineering vs. Deep Learning (CNNs)**

When people think of computer vision training taking hours or days, they are thinking of **Deep Learning / Convolutional Neural Networks (CNNs)** (such as ResNet, VGG, or YOLO):
* In Deep Learning, a neural network takes millions of raw pixel values ($256 \times 256 \times 3 = 196,608$ numbers per image) and runs forward/backward backpropagation loops over thousands of iterations to learn edge weights.
* **HOWEVER, your project specification (`project.md`, Line 93 & Line 771) strictly forbade Deep Learning:**
  > *"The contribution of Digital Image Processing must remain the central part of this project. Do not build a generic web application with only an AI API call. Do not add deep learning or CNNs."*

### How Our Pipeline Works Instead:
1. **The Heavy DIP Lifting (Feature Extraction):**  
   OpenCV and NumPy inspect each image mathematically and compute **8 summary numbers**:
   `[Brightness, Contrast, Sharpness, Noise, Area Ratio, Centroid Alignment, Background Uniformity, Boundary Margin]`  
   This step processes spatial pixels, runs 2D convolutions, computes histograms, and finds contours. This took ~15–20 seconds to process all 100 images.
2. **The ML Training Step:**  
   The dataset given to Scikit-learn's Random Forest is **not** raw pixels — it is a tabular spreadsheet (`dataset/extracted_features.csv`) with **80 rows and 8 columns**.
3. **Training Time for Random Forest on an $80 \times 8$ Table:**  
   Building 100 decision trees on a small matrix of 80 rows takes approximately **0.15 to 0.30 seconds** on a modern CPU.
   * It is **100% mathematically correct**, non-trivial, and standard textbook pattern recognition.
   * You can confidently explain to your professor: *"Sir, we followed the classical DIP paradigm: we extracted 8 physically meaningful DIP feature vectors using OpenCV first, and then trained a Random Forest ensemble on that structured tabular feature space."*

---

## 2. Master Requirement-to-Code Mapping

To make this easy to navigate and explain to your professor, every single requirement from [`project.md`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/project.md) is organized below into logical stages. Each stage shows the exact file, line location, and function implementing it:

---

### 🔹 Part A: Image Preprocessing & Low-Level Quality Analysis

#### Stage 1: Image Input & Preprocessing (`project.md` Lines 141–167)
* **Goal:** Standardize uploaded product images into consistent formats and dimensions without distorting the product.
* **Code File:** [`src/preprocessing.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/preprocessing.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Read image & BGR to RGB** | Matrix loading / Color correction | `load_image()` | Converts OpenCV's native BGR to RGB format |
| **RGB to Grayscale** | $Y = 0.299R + 0.587G + 0.114B$ | `to_grayscale()` | Perceptual luminance mapping (ITU-R BT.601) |
| **Aspect-ratio Resizing** | Bilinear / Area Interpolation | `resize_image()` | Limits max dimension to 600px without stretching |
| **Pixel Normalization** | $I_{norm} = I / 255.0$ | `normalize_image()` | Normalizes dynamic range to [0.0, 1.0] |
| **Pipeline Wrapper** | Full Stage 1 integration | `preprocess_pipeline()` | Returns RGB, Grayscale, and dimensions dict |

---

#### Stage 2: Brightness & Contrast Analysis (`project.md` Lines 168–208)
* **Goal:** Identify underexposed (dark), overexposed (washed out), and low-contrast product images.
* **Code File:** [`src/quality_analysis.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/quality_analysis.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **256-Bin Histogram** | Intensity frequency $h(r_k) = n_k$ | `compute_histogram()` | Analyzes distribution of pixel brightness |
| **Mean Brightness** | $\mu = \frac{1}{N} \sum I(x, y)$ | `calculate_brightness()` | Quantifies overall lighting level (0–255) |
| **Contrast Estimation** | $\sigma = \sqrt{\frac{1}{N} \sum (I - \mu)^2}$ | `calculate_contrast()` | Measures intensity spread (standard deviation) |
| **Histogram Equalization** | Cumulative Distribution (CDF) | `histogram_equalization()` | Stretches contrast across dynamic range |
| **Contrast Stretching** | Linear min-max percentile scaling | `contrast_stretching()` | Robust contrast expansion without outlier noise |
| **Quality Interpretation** | Configurable heuristic bounds | `analyze_brightness_contrast()` | Returns `GOOD` / `LOW` / `HIGH` status |

---

#### Stage 3: Sharpness & Blur Detection (`project.md` Lines 209–237)
* **Goal:** Detect out-of-focus optics and camera shake using spatial frequency response.
* **Code File:** [`src/sharpness.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/sharpness.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Laplacian Filtering** | $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$ | `compute_laplacian()` | 2nd-order spatial derivative with 64-bit precision |
| **Sharpness Score** | $\text{Var}(\nabla^2 I) = \frac{1}{N}\sum (L - \mu_L)^2$ | `calculate_sharpness_score()` | High variance = sharp; Low variance = blurry |
| **Edge Visualization** | 8-bit scale mapping $|L(x, y)|$ | `get_laplacian_display()` | Produces displayable edge response image |
| **Blur Detection** | Variance thresholding ($< 80$) | `analyze_sharpness()` | Returns `SHARP` or `BLURRY` diagnostic |

---

#### Stage 4: Noise Detection & Filtering (`project.md` Lines 238–262)
* **Goal:** Quantify sensor grain and demonstrate linear vs. non-linear noise reduction.
* **Code File:** [`src/noise.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/noise.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Noise Estimation** | Immerkær (1996) residual filter | `estimate_noise()` | Measures noise standard deviation $\sigma_n$ directly |
| **Gaussian Smoothing** | 2D Gaussian bell kernel | `apply_gaussian_filter()` | Linear spatial averaging for Gaussian noise |
| **Median Filtering** | Non-linear order-statistic median | `apply_median_filter()` | Completely eliminates impulse salt-and-pepper noise |
| **Synthetic Noise** | $N(0, \sigma^2)$ & Random impulse | `add_gaussian_noise()`, `add_salt_and_pepper_noise()` | Injects controlled noise for benchmarking |

---

### 🔹 Part B: Segmentation, Morphology & Layout Compliance

#### Stage 5: Product Segmentation (`project.md` Lines 263–292)
* **Goal:** Separate the product foreground from the studio background.
* **Code File:** [`src/segmentation.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/segmentation.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Polarity Detection** | Margin corner pixel sampling | `estimate_background_intensity()` | Checks if background is studio white or dark |
| **Otsu Thresholding** | Maximizes $\sigma_B^2(t) = \omega_0 \omega_1 (\mu_0 - \mu_1)^2$ | `segment_otsu()` | Automatic optimal bi-modal threshold calculation |
| **GrabCut Fallback** | Markov Random Field (MRF) Graph-Cut | `segment_grabcut()` | Recovers foreground when simple thresholding fails |
| **Studio Cutouts** | Bitwise masking operations | `extract_foreground()`, `extract_white_background()` | Renders isolated product on black and white studio BG |
| **Reliability Check** | Area ratio bounds ($4\% - 96\%$) | `assess_segmentation_reliability()` | Flags `RELIABLE` or `UNRELIABLE` status |

---

#### Stage 6: Morphological Mask Refinement (`project.md` Lines 293–319)
* **Goal:** Clean up noise specks and interior holes in the binary segmentation mask.
* **Code File:** [`src/morphology.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/morphology.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Structuring Element** | Kernel matrix (Rectangle / Ellipse) | `get_structuring_element()` | Probing geometric shape for morphological ops |
| **Morphological Opening** | $A \circ B = (A \ominus B) \oplus B$ | `apply_opening()` | Erosion followed by dilation; removes stray salt noise |
| **Morphological Closing** | $A \bullet B = (A \oplus B) \ominus B$ | `apply_closing()` | Dilation followed by erosion; fills interior holes |
| **Mask Cleaning** | Opening $\rightarrow$ Closing pipeline | `clean_segmentation_mask()` | Returns final refined binary mask |

---

#### Stage 7: Object & Layout Analysis (`project.md` Lines 320–354)
* **Goal:** Verify that the product is properly centered, occupies sufficient space, and is not accidentally cropped.
* **Code File:** [`src/object_analysis.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/object_analysis.py)

| Operation from `project.md` | DIP Concept / Math | Function Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Contour Detection** | Topological curve extraction | `find_primary_contour()` | Finds boundary of primary product |
| **Centroid Moments** | $C_x = m_{10}/m_{00}, C_y = m_{01}/m_{00}$ | `calculate_centroid()` | Computes spatial center of mass |
| **Centering Alignment** | Normalized Euclidean distance | `calculate_alignment_score()` | Produces score (0.0 to 1.0) measuring alignment |
| **Boundary Proximity** | Edge margin distance check | `check_boundary_proximity()` | Flags warning if product touches canvas edges |
| **Background Uniformity** | Std dev of non-product pixels | `calculate_background_uniformity()` | Verifies clean solid studio background |
| **Visual Annotations** | Green BBox & Red Centroid (+) | `draw_object_annotations()` | Visual overlay for dashboard display |

---

### 🔹 Part C: Feature Engineering & Dataset Pipeline

#### Stage 8: Feature Vector Extraction (`project.md` Lines 356–386)
* **Goal:** Aggregate all DIP calculations into an 8-dimensional numerical vector for machine learning.
* **Code File:** [`src/feature_extraction.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/feature_extraction.py)

| Output Feature | Source DIP Stage | Range | Significance |
| :--- | :--- | :--- | :--- |
| `brightness` | Stage 2 | $0 - 255$ | Detects underexposure and overexposure |
| `contrast` | Stage 2 | $0 - 128$ | Detects washed out or flat images |
| `sharpness` | Stage 3 | $0 - 50,000+$ | Detects blurry vs crisp images |
| `noise` | Stage 4 | $0 - 40+$ | Detects electronic sensor grain |
| `object_area_ratio`| Stage 7 | $0.0 - 1.0$ | Verifies product framing size |
| `alignment_score` | Stage 7 | $0.0 - 1.0$ | Verifies centered positioning |
| `bg_uniformity` | Stage 7 | $0.0 - 1.0$ | Verifies clean solid backdrop |
| `boundary_margin` | Stage 7 | $0.0 - 0.5$ | Warns against accidental cropping |

---

#### Section 7: Dataset Curation & Defect Synthesis (`project.md` Lines 387–444)
* **Goal:** Create a reproducible, zero-leakage benchmark dataset from real Amazon products.
* **Code File:** [`data_generator.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/data_generator.py)

| Task | Implementation | Traceability |
| :--- | :--- | :--- |
| **Amazon Sourcing** | Filtered clean studio products from **Amazon Berkeley Objects (ABO)** | Stored in `dataset/raw/` |
| **Zero Data Leakage** | Product IDs partitioned **80% Train / 20% Test** *before* creating defects | Documented in `dataset/manifest.csv` |
| **Defect Synthesis** | Gaussian blur, motion blur, underexposure, overexposure, low contrast, Gaussian noise, salt-and-pepper | Saved to `dataset/train/` and `dataset/test/` |

---

### 🔹 Part D: Decision Systems & Benchmarking

#### Section 8: Traditional Rule-Based Baseline Engine (`project.md` Lines 446–481)
* **Goal:** Traditional heuristic benchmark applying fixed 1D threshold cutoffs with verbal explanations.
* **Code File:** [`src/baseline.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/baseline.py)
* **Function:** `evaluate_rules(features, thresholds)`
* **Output:** Returns `ACCEPT` / `REJECT` along with explicit textual reasons (e.g. *"Image is blurry (Sharpness: 12.1 < 80.0)"*).

---

#### Section 9: Random Forest Machine Learning Model (`project.md` Lines 483–518)
* **Goal:** Multi-dimensional statistical pattern classifier learning non-linear feature interactions.
* **Code File:** [`src/classifier.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/classifier.py)
* **Functions:**
  - `train_random_forest()`: Trains 100 decision trees with `class_weight='balanced'`.
  - `evaluate_classifier()`: Calculates Accuracy, Precision, Recall, F1, FRR, and Confusion Matrix.
  - `predict_single()`: Returns classification and prediction probability confidence.
  - `save_model()`, `load_model()`: Serializes model to [`models/random_forest.joblib`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/models/random_forest.joblib).

---

#### Section 10: Quantitative Comparison Pipeline (`project.md` Lines 520–559)
* **Goal:** Evaluates Baseline and Random Forest on the **exact same held-out test images**.
* **Code File:** [`train.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/train.py)
* **Output Files Generated:**
  - Tabular comparison: [`results/comparisons/baseline_vs_random_forest.csv`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/results/comparisons/baseline_vs_random_forest.csv)
  - Confusion matrix plot: [`results/plots/confusion_matrices.png`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/results/plots/confusion_matrices.png)
  - Full metrics JSON: [`results/metrics/evaluation_summary.json`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/results/metrics/evaluation_summary.json)

---

### 🔹 Part E: Dashboard & Automated Verification

#### Section 11: Interactive Streamlit Dashboard (`project.md` Lines 560–641)
* **Goal:** Visual web interface for real-time image inspection and viva demonstration.
* **Code File:** [`app.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/app.py)
* **Contains all required sections:**
  - **Section A:** Original Image, dimensions, aspect ratio, dynamic range.
  - **Section B:** Real-time metrics for Brightness, Contrast, Sharpness, Noise, Area %.
  - **Section C:** Grayscale, Histogram Equalization, Median Denoised, Laplacian Edges.
  - **Section D & E:** Binary mask, isolated product cutouts, BBox & Centroid (+).
  - **Section F:** Prominent **ACCEPT / REJECT** banner comparing Baseline vs. ML.
  - **Academic Benchmark Tab:** Displays quantitative comparison table and confusion matrices.

---

#### Section 13: 6 Quality Condition Test Suites (`project.md` Lines 694–739)
* **Goal:** Automated verification of the 6 mandatory image conditions specified in the project guidelines.
* **Code File:** [`tests/test_end_to_end.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/tests/test_end_to_end.py)
* **Function:** `run_all_test_cases()` — Validates Good, Blurry, Dark, Noisy, Irregular Background, and Segmentation fallback edge cases.
* **Central Test Runner:** [`run_tests.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/run_tests.py) — Runs all 10 unit test files in a single command.

---

## 3. End-to-End Pipeline Walkthrough

Here is the exact data flow that runs when an image is uploaded or processed:

```
[ Uploaded Amazon Product Image ]
                │
                ▼
      ┌──────────────────┐
      │  PREPROCESSING   │ ──► [Resized RGB (max 600px)] & [8-bit Grayscale]
      └─────────┬────────┘
                │
    ┌───────────┼───────────────┬────────────────┐
    ▼           ▼               ▼                ▼
[BRIGHTNESS] [SHARPNESS]     [NOISE]       [SEGMENTATION]
  Mean μ       Var(∇²I)     Immerkær σ_n     Otsu + GrabCut
  Std Dev σ    Blur Flag    Median Filter    Opening & Closing
    │           │               │                │
    └───────────┼───────────────┴────────────────┘
                ▼
      ┌──────────────────┐
      │ OBJECT & LAYOUT  │ ──► Centroid (Cx, Cy), Area Ratio %,
      │     ANALYSIS     │     Boundary Cropping Margin, Bg Uniformity
      └─────────┬────────┘
                ▼
      ┌──────────────────┐
      │ 8-D FEATURE ROW  │ ──► [Brightness, Contrast, Sharpness, Noise,
      └─────────┬────────┘      Area, Alignment, Uniformity, Margin]
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌───────────────────────┐
│ BASELINE DIP  │ │ RANDOM FOREST ML      │
│ (Fixed Rules) │ │ (Learned Boundaries)  │
└───────┬───────┘ └───────────┬───────────┘
        │                     │
        └───────────┬─────────┘
                    ▼
          ┌──────────────────┐
          │ STREAMLIT UI     │
          │ Prominent Result │
          │ Visual Artifacts │
          │ Side-by-Side     │
          └──────────────────┘
```

---

## 4. File & Module Architecture Guide

```
dip_project/
│
├── dataset/
│   ├── manifest.csv               # Master log of all 100 images, splits, and defect types
│   ├── extracted_features.csv     # Extracted 8-D numerical DIP table for all 100 images
│   ├── raw/                       # 100 base Amazon product images
│   ├── train/                     # 80 training images (40 ACCEPT clean, 40 REJECT defective)
│   └── test/                      # 20 held-out test images (10 ACCEPT clean, 10 REJECT defective)
│
├── src/                           # Pure Python DIP and ML library modules
│   ├── preprocessing.py           # Color-spaces, aspect-ratio scaling, normalization
│   ├── quality_analysis.py        # Brightness, contrast, histogram equalization
│   ├── sharpness.py               # Laplacian variance blur detection
│   ├── noise.py                   # Immerkær noise estimation, Gaussian & median filters
│   ├── segmentation.py            # Otsu thresholding with GrabCut fallback
│   ├── morphology.py              # Mathematical morphology (Opening & Closing)
│   ├── object_analysis.py         # Image moments, centroid, bounding box, crop warnings
│   ├── feature_extraction.py      # Combines Stages 1-7 into single feature vector
│   ├── baseline.py                # Deterministic rule engine with verbal reasons
│   └── classifier.py              # Random Forest training, evaluation, saving/loading
│
├── models/
│   └── random_forest.joblib       # Serialized trained Random Forest classifier
│
├── notebooks/                     # Step-by-step interactive Jupyter notebooks
│   ├── 01_preprocessing.ipynb    # Visual demo of Stage 1
│   ├── 02_quality_analysis.ipynb # Visual demo of Stages 2, 3, 4
│   ├── 03_segmentation.ipynb     # Visual demo of Stages 5, 6, 7
│   └── 04_classification.ipynb   # Visual demo of ML benchmarking
│
├── results/
│   ├── comparisons/               # baseline_vs_random_forest.csv
│   ├── metrics/                   # evaluation_summary.json
│   └── plots/                     # confusion_matrices.png, feature_distributions.png, pipeline_walkthrough.png
│
├── tests/                         # Unit tests for all modules
│   ├── test_preprocessing.py
│   ├── test_quality_analysis.py
│   ├── test_sharpness.py
│   ├── test_noise.py
│   ├── test_segmentation.py
│   ├── test_object_analysis.py
│   ├── test_feature_extraction.py
│   ├── test_baseline.py
│   ├── test_classifier.py
│   └── test_end_to_end.py         # Validates all 6 mandatory test cases
│
├── app.py                         # Interactive Streamlit dashboard
├── train.py                       # Extracts features & trains Random Forest model
├── data_generator.py              # Downloads & filters ABO dataset, generates defects
├── run_tests.py                   # Central test runner executing all 10 test modules
├── requirements.txt               # Locked dependencies
└── README.md                      # This document
```

---

## 5. DIP Mathematical Formulas Implemented

### 1. Grayscale Conversion (ITU-R BT.601)
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
*Code:* [`src/preprocessing.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/preprocessing.py) (`to_grayscale()`)

### 2. Brightness ($\mu$) & Contrast ($\sigma$)
$$\mu = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y), \quad \sigma = \sqrt{\frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} (I(x, y) - \mu)^2}$$
*Code:* [`src/quality_analysis.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/quality_analysis.py) (`calculate_brightness()`, `calculate_contrast()`)

### 3. Sharpness Estimation (Variance of the Laplacian)
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
$$\text{Sharpness Score} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x,y} (L(x, y) - \mu_L)^2$$
*Code:* [`src/sharpness.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/sharpness.py) (`calculate_sharpness_score()`)

### 4. Noise Estimation (Immerkær 1996 High-Pass Residual)
$$\sigma_n = \sqrt{\frac{\pi}{2}} \cdot \frac{1}{6(W - 2)(H - 2)} \sum_{x,y} |I * M|$$
$$M = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}$$
*Code:* [`src/noise.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/noise.py) (`estimate_noise()`)

### 5. Otsu's Optimal Between-Class Variance
$$\sigma_B^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2$$
*Code:* [`src/segmentation.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/segmentation.py) (`segment_otsu()`)

### 6. Spatial Image Moments & Centroid
$$m_{p,q} = \sum_{x,y} x^p y^q I(x, y) \implies C_x = \frac{m_{10}}{m_{00}}, \quad C_y = \frac{m_{01}}{m_{00}}$$
*Code:* [`src/object_analysis.py`](file:///c:/Users/aryan/OneDrive/Desktop/dip_project/src/object_analysis.py) (`calculate_centroid()`)

---

## 6. Quantitative Benchmark: Baseline vs. Random Forest

Both models were evaluated on the **exact same held-out test dataset** (20 unseen Amazon Berkeley Objects product images: 10 Good, 10 Defective across Blur, Motion Blur, Lighting, Low-Contrast, and Noise):

| Metric | Baseline (Rule-Based DIP) | Improved (Random Forest) | Real-World Impact |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **65.00%** | **80.00%** | +15.0% overall diagnostic reliability |
| **Precision** | 60.00% | **80.00%** | Higher confidence defect alarms |
| **Recall** | **90.00%** | **80.00%** | Catches the vast majority of defective uploads |
| **F1 Score** | 0.7200 | **0.8000** | Superior balanced performance |
| **False Reject Rate (FRR)** | **60.00%** | **20.00%** | **Cuts wrongful seller rejections by 3x** |

### Confusion Matrix Breakdown:
* **Baseline (Rule-Based):** Correctly accepted 4 clean images, but wrongly rejected 6 clean images (60% FRR).
* **Random Forest ML:** Correctly accepted 8 clean images, wrongly rejecting only 2 (20% FRR).

---

## 7. How to Present & Demonstrate This to Your Professor

When your professor asks you to demonstrate the project, follow this 4-step walkthrough:

### Step 1: Open the Interactive Dashboard
Run in your terminal:
```bash
streamlit run app.py
```
* Point out: *"Sir, this is our Automated Image Quality Inspector running locally with a clean web interface."*

### Step 2: Demonstrate Preset Test Cases (Sidebar)
* Select **"Preset Academic Test Cases"** from the left sidebar.
* **Select Sample 1 (Good Product):** Show that both Baseline and Random Forest output **ACCEPT** with high confidence, showing crisp edges and high sharpness.
* **Select Sample 2 (Blurry Product):** Show how the Sharpness score collapses to ~12.1 (below threshold 80), triggering the **REJECT** decision with the explanation: *"Image is blurry"*.
* **Select Sample 3 (Dark Product):** Show how Mean Brightness drops to 36.5 (below 55), triggering **REJECT** for underexposure.
* **Select Sample 5 (High Noise):** Show how the Immerkær noise estimate jumps above 13.5, triggering **REJECT** for excessive noise, and show the **Median Filtered** image in Section C that cleaned it up.

### Step 3: Show the Segmentation & Morphological Output (Sections D & E)
* Scroll down to Section D & E.
* Show the **Morphological Binary Mask**, the **White Studio Background Cutout**, and the green **Bounding Box with red Centroid (+)** crosshair.
* Explain: *"Sir, we use spatial image moments $m_{10}/m_{00}$ to verify if the product is centered and check boundary margins to warn against accidental cropping."*

### Step 4: Show the Quantitative Benchmark Tab
* In the sidebar, click **"Model Benchmark Report"**.
* Show the comparison table and confusion matrix plot:
  * *"Sir, here is the quantitative comparison on the held-out test set. Fixed rules have a 60% False Reject Rate because if an image misses any single threshold, it gets rejected. Random Forest learns multivariate relationships across all 8 DIP features, reducing false rejects to 20% and achieving 80% overall accuracy."*

---

## 8. Verification & How to Run Everything

### Run the Central Test Suite (Validates all 10 modules in ~5 seconds)
```bash
python run_tests.py
```
Expected output:
```
======================================================================
TEST SUMMARY: 10 PASSED, 0 FAILED (Total: 10)
======================================================================
ALL TESTS COMPLETED SUCCESSFULLY!
```

### Run the Automated End-to-End Validation (Validates all 6 test conditions)
```bash
python -m tests.test_end_to_end
```

### Launch the Streamlit Inspector
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
