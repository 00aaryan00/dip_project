# PROJECT: AUTOMATED E-COMMERCE IMAGE QUALITY AND COMPLIANCE INSPECTOR

## 1. YOUR ROLE

Act as a Senior Computer Vision Engineer, Digital Image Processing Engineer, Machine Learning Engineer and Python Developer.

I want you to help me develop a complete academic Digital Image Processing (DIP) course project from scratch.

You are my primary coding and development assistant.

Your responsibilities:

* Understand the complete project before coding.
* Design the architecture.
* Create a clean project structure.
* Implement each module independently.
* Explain important technical decisions.
* Test the implementation.
* Integrate all modules into one working application.
* Prepare evaluation results and documentation.

Do not just generate code without understanding the complete workflow.

I am a B.Tech Information Technology student. I understand basic programming and want to understand the implementation sufficiently to explain it during my project demonstration and viva.

Use simple, readable Python code with proper comments.

---

# 2. PROJECT BACKGROUND

Project Title:

Automated E-commerce Image Quality and Compliance Inspector

Reference Project ID: HSR-AIML-056

Industry Reference: Neuropixel.ai

Domain: Digital Image Processing, Computer Vision, Machine Learning and E-commerce.

## Problem Statement

E-commerce platforms receive thousands of product images from different sellers.

However, many uploaded images contain quality-related problems:

1. Blurry product images.
2. Poor brightness or underexposure.
3. Low contrast.
4. Excessive image noise.
5. Improper or inconsistent backgrounds.
6. Product not properly positioned.
7. Product occupying too little or too much image space.
8. Product partially cropped or poorly visible.

Manually inspecting every uploaded image is time-consuming.

Our objective is to develop an automated image inspection system that analyses uploaded product images using Digital Image Processing techniques and Machine Learning.

The system should identify image quality defects, calculate meaningful image features and classify images as:

* ACCEPT
* REJECT

For rejected images, the system should also provide understandable reasons.

IMPORTANT:

This is an academic prototype. We are implementing basic visual quality and layout compliance checks, not complete compliance with every marketplace's commercial policies.

---

# 3. MAIN PROJECT OBJECTIVES

The system must:

1. Accept product images as input.
2. Preprocess images using OpenCV.
3. Analyse brightness and contrast.
4. Detect blur using sharpness estimation.
5. Estimate image noise.
6. Enhance images where appropriate.
7. Segment product foreground from background.
8. Apply morphological operations.
9. Analyse product position, size and background.
10. Extract numerical image features.
11. Implement a traditional rule-based baseline.
12. Implement an improved Random Forest classifier.
13. Compare both approaches quantitatively.
14. Display all results through a simple Streamlit dashboard.

The contribution of Digital Image Processing must remain the central part of this project.

Do not build a generic web application with only an AI API call.

---

# 4. TECHNOLOGY STACK

Use the following technologies:

Programming Language:

* Python 3

Core Libraries:

* OpenCV (cv2)
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Pillow

Application:

* Streamlit

Evaluation:

* Scikit-learn metrics
* Confusion Matrix
* Classification Report

Development Environment:

* Google Antigravity IDE
* VS Code compatible project structure

Do not use React, Node.js, MongoDB, cloud infrastructure or complex backend architecture.

The project must run locally.

---

# 5. COMPLETE DIGITAL IMAGE PROCESSING PIPELINE

Implement the following stages.

## STAGE 1: IMAGE INPUT AND PREPROCESSING

Input:
An uploaded product image.

Operations:

1. Read image using OpenCV.
2. Convert BGR to RGB where necessary.
3. Convert RGB to grayscale.
4. Resize images to a consistent resolution where appropriate.
5. Normalize pixel intensity if required.
6. Preserve the original image for comparison.

Display:

* Original image.
* Grayscale image.
* Resized image.

Concepts:

* Image representation.
* RGB and grayscale conversion.
* Image resizing.
* Pixel intensity normalization.

## STAGE 2: BRIGHTNESS AND CONTRAST ANALYSIS

Objective:
Identify whether an image is too dark, too bright or has poor contrast.

Implement:

1. Image histogram.
2. Mean pixel intensity.
3. Standard deviation.
4. Contrast estimation.
5. Histogram Equalization.
6. Contrast Stretching.

Calculate:

Mean Brightness:

Mean = Sum of all pixel intensities / Total number of pixels.

Contrast:

Use grayscale standard deviation as a basic contrast indicator.

Also calculate histogram-based information such as intensity distribution.

Show:

* Original grayscale image.
* Original histogram.
* Enhanced image.
* Enhanced histogram.
* Brightness score.
* Contrast score.

Provide an interpretation:

GOOD / LOW / HIGH

Do not assume one universal threshold works for every product category. Keep thresholds configurable.

## STAGE 3: SHARPNESS AND BLUR DETECTION

Objective:
Identify blurry product images.

Use:

1. Laplacian operator.
2. Laplacian variance.
3. Edge analysis.

Formula:

Sharpness Score = Variance(Laplacian(Grayscale Image))

Interpretation:

* Low variance: Potentially blurry image.
* High variance: More visible high-frequency details.

Display:

* Original image.
* Laplacian response image.
* Sharpness score.
* Blur detection result.

Use configurable thresholds and explain their limitations.

## STAGE 4: IMAGE NOISE DETECTION AND FILTERING

Objective:
Estimate noise in images and demonstrate noise reduction.

Implement:

1. Gaussian noise generation.
2. Salt-and-pepper noise generation.
3. Gaussian filtering.
4. Median filtering.
5. Noise estimation or a clearly documented noise proxy.

Compare:

Original image.
Noisy image.
Gaussian filtered image.
Median filtered image.

Explain why median filtering is useful for salt-and-pepper noise and Gaussian filtering is useful for smoothing Gaussian-like noise.

Important:
Do not claim that a denoising filter improves every image. Show the actual before/after results.

## STAGE 5: PRODUCT SEGMENTATION

Objective:
Separate product foreground from the background.

Implement classical segmentation techniques:

1. Thresholding.
2. Binary thresholding.
3. Otsu thresholding.
4. Foreground mask generation.

For simple backgrounds, use thresholding-based segmentation.

Where necessary, use OpenCV GrabCut as an alternative.

Generate:

* Original image.
* Binary segmentation mask.
* Foreground product extraction.
* Background visualization.

Important:
Simple thresholding does not work reliably for every product and background combination.

Implement sensible fallback behaviour and clearly identify when segmentation is unreliable.

Do not claim perfect background removal.

## STAGE 6: MORPHOLOGICAL OPERATIONS

Apply morphological operations to the binary segmentation mask.

Implement:

1. Erosion.
2. Dilation.
3. Opening.
4. Closing.

Explain their purpose:

Erosion:
Shrinks foreground regions.

Dilation:
Expands foreground regions.

Opening:
Removes small unwanted foreground regions.

Closing:
Fills small holes and gaps.

Display intermediate results of these operations.

## STAGE 7: OBJECT AND BACKGROUND ANALYSIS

After segmentation, analyse the product region.

Implement:

1. Contour detection.
2. Connected component analysis.
3. Bounding box extraction.
4. Object area calculation.
5. Centroid calculation.
6. Object-to-image area ratio.
7. Basic centre alignment estimation.
8. Background uniformity estimation.
9. Boundary proximity check for possible cropping.

Calculate features such as:

Object Area Ratio =
Product Foreground Area / Total Image Area

Use bounding box and centroid to estimate product placement.

For cropping, use boundary proximity only as a warning signal, not as definitive proof that a product is cropped.

Display:

* Segmented product.
* Bounding box.
* Product centroid.
* Object area percentage.
* Background uniformity.
* Position analysis.

---

# 6. FEATURE EXTRACTION

Combine all relevant measurements into a structured feature vector.

Each image should generate numerical features such as:

1. Mean Brightness.
2. Contrast Score.
3. Laplacian Variance.
4. Noise Estimate.
5. Object Area Ratio.
6. Centre Alignment Score.
7. Background Uniformity Score.
8. Boundary Proximity Score.

Store extracted features in a Pandas DataFrame.

Example:

| Brightness | Contrast | Sharpness | Noise | Object Area | Background | Label |
| ---------- | -------- | --------- | ----- | ----------- | ---------- | ----- |
| 125        | 55       | 140       | 12    | 0.65        | 0.95       | Good  |
| 35         | 20       | 110       | 15    | 0.60        | 0.90       | Bad   |
| 130        | 60       | 25        | 18    | 0.70        | 0.95       | Bad   |

These values are illustrative.

Actual feature values must come from image processing.

---

# 7. DATASET PREPARATION

We need a dataset containing product images.

Explore suitable public product image datasets such as:

1. Amazon Berkeley Objects (ABO).
2. Fashion Product Images Dataset.
3. DeepFashion.

Choose a manageable dataset that can be obtained and processed quickly.

Do not spend excessive time downloading huge datasets.

Start with approximately 100–200 original product images if feasible.

Create synthetic defects from original images.

Generate variations:

* Low brightness.
* High brightness.
* Low contrast.
* Gaussian blur.
* Motion blur.
* Gaussian noise.
* Salt-and-pepper noise.

Organize the data clearly.

Suggested structure:

dataset/
original/
good/
defective/
blurry/
dark/
low_contrast/
noisy/

Generate labels systematically.

Maintain traceability between original images and their modified versions.

IMPORTANT DATASET RULE:

Do not put different augmented versions of the same original image into both training and testing datasets.

Split the original images into training and testing groups first, then generate or assign their augmented variants within their respective groups.

This prevents data leakage.

Keep the dataset manageable because the project must be completed within 4–5 days.

Do not invent dataset statistics or claim that synthetic data represents real-world marketplace distributions.

---

# 8. BASELINE APPROACH — RULE-BASED CLASSIFICATION

Implement the first classification approach using traditional Digital Image Processing.

Use manually configurable thresholds.

Example:

If sharpness < threshold:
Reject due to blur.

If brightness < threshold:
Reject due to low brightness.

If noise > threshold:
Reject due to excessive noise.

If background uniformity < threshold:
Flag background issue.

Otherwise:
Accept.

The exact thresholds should be calibrated using the dataset.

Output:

ACCEPT / REJECT

Also generate defect reasons.

This approach will be our BASELINE MODEL.

Do not use machine learning in this stage.

---

# 9. IMPROVED APPROACH — RANDOM FOREST CLASSIFICATION

Implement a Machine Learning classifier.

Use:

RandomForestClassifier from Scikit-learn.

Input:
Extracted numerical image features.

Output:
Good Quality / Poor Quality.

Steps:

1. Load the feature dataset.
2. Separate features and labels.
3. Perform train-test split without data leakage.
4. Train Random Forest.
5. Make predictions.
6. Calculate evaluation metrics.
7. Save the trained model using joblib.
8. Load the trained model during dashboard inference.

Use a reproducible random seed.

Use class_weight='balanced' if the dataset is imbalanced and justify its use.

Do not train the model on arbitrary fabricated labels.

Ensure training labels are based on documented image annotations or known defect-generation rules.

The baseline and ML model must use the same held-out test images for a fair comparison.

---

# 10. BASELINE VS IMPROVED COMPARISON

This comparison is an essential requirement of the DIP course.

Baseline:
Traditional DIP + Fixed Rules.

Improved:
Traditional DIP + Feature Extraction + Random Forest.

Compare both approaches using:

1. Accuracy.
2. Precision.
3. Recall.
4. F1 Score.
5. Confusion Matrix.
6. False Reject Rate.

Show a comparison table.

Example format:

| Metric    | Baseline      | Improved      |
| --------- | ------------- | ------------- |
| Accuracy  | Actual Result | Actual Result |
| Precision | Actual Result | Actual Result |
| Recall    | Actual Result | Actual Result |
| F1 Score  | Actual Result | Actual Result |

IMPORTANT:

Never fabricate performance numbers.

Calculate all metrics from actual model predictions.

If the Random Forest model does not outperform the baseline, report the actual result and explain why.

---

# 11. FINAL STREAMLIT DASHBOARD

Create a clean, professional and easy-to-understand interface.

Application title:

Automated E-commerce Image Quality & Compliance Inspector

The dashboard must allow the user to upload an image.

After upload, display the following sections.

SECTION A — ORIGINAL IMAGE

* Uploaded image.
* Image dimensions.
* File format.

SECTION B — IMAGE QUALITY ANALYSIS

* Brightness score.
* Contrast score.
* Sharpness score.
* Noise estimate.
* Individual quality status.

SECTION C — IMAGE ENHANCEMENT

* Original image.
* Histogram.
* Enhanced image.
* Filtered image.

SECTION D — PRODUCT SEGMENTATION

* Binary mask.
* Extracted foreground.
* Bounding box.
* Object area ratio.

SECTION E — BACKGROUND AND OBJECT ANALYSIS

* Background uniformity.
* Product position.
* Centre alignment.
* Boundary proximity warning.

SECTION F — FINAL CLASSIFICATION

Show a prominent result:

ACCEPT or REJECT.

Also display:

* Predicted quality class.
* Defect reasons.
* Individual feature scores.
* Recommendations.

For example:

IMAGE QUALITY REPORT

Brightness: GOOD
Contrast: GOOD
Sharpness: POOR
Noise: MODERATE
Background: GOOD

FINAL RESULT: REJECT

Reason:
Image appears blurry due to low sharpness.

Show both baseline and Random Forest predictions where appropriate.

Clearly distinguish measured image features, rule-based warnings and ML predictions.

Do not imply that the dashboard has improved an image merely because it has analysed it.

---

# 12. PROJECT ARCHITECTURE AND FILE STRUCTURE

Create a modular project.

Suggested structure:

ecommerce-image-quality/
│
├── dataset/
│   ├── original/
│   ├── good/
│   └── defective/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_quality_analysis.ipynb
│   ├── 03_segmentation.ipynb
│   └── 04_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── quality_analysis.py
│   ├── enhancement.py
│   ├── noise_detection.py
│   ├── segmentation.py
│   ├── morphology.py
│   ├── object_analysis.py
│   ├── feature_extraction.py
│   ├── baseline.py
│   └── classifier.py
│
├── models/
│   └── random_forest.pkl
│
├── results/
│   ├── plots/
│   ├── comparisons/
│   └── metrics/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore

You may simplify or modify this structure if necessary, but maintain clear separation between image processing, model training, inference and UI.

Do not create unnecessary files or empty modules.

---

# 13. TESTING REQUIREMENTS

Test the project using different image conditions:

Test Case 1:
Good quality product image.

Expected:
ACCEPT.

Test Case 2:
Very blurry product image.

Expected:
Blur warning and likely REJECT.

Test Case 3:
Very dark product image.

Expected:
Low brightness warning.

Test Case 4:
Noisy product image.

Expected:
Noise warning.

Test Case 5:
Product image with an irregular background.

Expected:
Background analysis warning.

Test Case 6:
Product image with poor segmentation.

Expected:
Segmentation reliability warning.

The exact final classification should depend on calibrated rules and model predictions.

Do not force every example to produce a predetermined result.

---

# 14. FINAL ACADEMIC DELIVERABLES

The final project must include:

1. Complete working Python source code.
2. Dataset description and source.
3. Image preprocessing implementation.
4. Intermediate image processing results.
5. DIP pipeline block diagram.
6. Feature extraction implementation.
7. Baseline classifier.
8. Random Forest classifier.
9. Quantitative comparison.
10. Confusion matrix.
11. Streamlit application.
12. Screenshots of successful and failed cases.
13. README with execution instructions.
14. Final project report.
15. Presentation-ready results.

Keep the focus on Digital Image Processing concepts.

---

# 15. DEVELOPMENT CONSTRAINTS — VERY IMPORTANT

I have only 4–5 days to complete this project.

Therefore:

1. Prioritize a working end-to-end project.
2. Do not overengineer.
3. Do not add deep learning or CNNs.
4. Do not introduce unnecessary APIs.
5. Do not build React or Node.js applications.
6. Avoid unnecessarily large datasets.
7. Use lightweight Python libraries.
8. Keep code readable and modular.
9. Avoid complex abstractions.
10. Test every major module before integration.

If any feature is too time-consuming, implement a simpler academically valid alternative and document the limitation.

Do not sacrifice the core DIP implementation for UI design.

---

# 16. HOW I WANT YOU TO WORK

Follow these development stages.

PHASE 1:
Inspect the current workspace.

Identify:

* Existing files.
* Python installation.
* Available environment.
* Existing dependencies.

Do not overwrite useful existing files without checking them.

PHASE 2:
Create a detailed implementation plan.

Explain:

* Overall architecture.
* Module responsibilities.
* Data flow.
* Dependencies.
* Development sequence.

PHASE 3:
Implement the project incrementally.

For every module:

1. Explain what it does.
2. Explain which DIP concept it implements.
3. Write the code.
4. Run and test the code.
5. Fix errors.
6. Show expected outputs.

PHASE 4:
Integrate all modules into the main application.

PHASE 5:
Test the complete end-to-end workflow.

PHASE 6:
Generate README and final documentation.

Do not just create all files and declare the project complete.

Run the actual code and verify that the application works.

---

# 17. IMPORTANT LEARNING REQUIREMENT

I also need to understand the project for my academic viva.

Whenever implementing an important concept, briefly explain:

* What problem does this technique solve?
* Why are we using it?
* How does it work?
* What is its mathematical intuition?
* What is its output?
* How does it contribute to our project?

Use simple explanations.

Do not overwhelm me with unnecessary theory.

---

# 18. FIRST TASK — DO NOT START CODING IMMEDIATELY

Start by doing the following:

1. Understand the entire project.
2. Summarize the project in your own words.
3. Explain the complete processing pipeline.
4. Show the proposed folder structure.
5. Explain the role of every module.
6. Provide the exact development sequence for 4–5 days.
7. Identify technical risks and simplifications.

Then begin implementation with Phase 1.

Work incrementally and keep the project runnable after each major phase.

FINAL GOAL:

A complete, working Digital Image Processing academic project that accepts an e-commerce product image, analyses its visual quality using classical DIP techniques, extracts meaningful features, classifies the image using both traditional rules and Random Forest, compares both approaches quantitatively, and displays an understandable quality report through Streamlit.
