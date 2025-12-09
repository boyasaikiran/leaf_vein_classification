🌿 Leaf Vein Classification System using Machine Learning
📝 Project Summary

This project implements an Automated Leaf Vein Classification System using computer vision (OpenCV) and machine learning (Random Forest). The core idea is that the vein structure (venation) of a plant leaf contains unique features that can be used to accurately identify both the species and its health condition (healthy vs. diseased).

The system processes an uploaded leaf image through a structured pipeline: Preprocessing → Feature Extraction → Machine Learning Classification.
🔬 Methodology
1. 🧮 Preprocessing & Skeletonization (OpenCV)

The initial image is transformed into a binary skeleton map of the veins, isolating the structural pattern for analysis.

The key steps are:

    Image Preparation: Resize and normalize the input image.

    Grayscale Conversion: Convert the image to grayscale.

    Noise Reduction: Apply a Gaussian filter.

    Segmentation: Apply Binary thresholding.

    Vein Extraction: Use Morphological thinning (skeletonization) to obtain a single-pixel-wide vein map.

2. 🧪 Feature Extraction

From the skeletonized vein map, quantitative, structural, and statistical metrics are calculated.

The Output is a 13-dimensional feature vector for each leaf, which includes:

    Structural Metrics: Number of vein segments, total length of veins, mean branch thickness.

    Density Metrics: Area covered by veins, vein density metrics.

    Statistical Descriptors: Additional statistical measures of the vein pattern.

3. 🌲 Model Used: RandomForestClassifier

A Random Forest Classifier is used for the prediction task. This model is chosen for its robustness, ability to handle noise, and effectiveness with structured features typical of biological patterns.

    Training Setup:

        Train/Test Split: 80% / 20%

        Normalization: Applied using StandardScaler.

        Model Artefacts: Trained model and scaler saved as random_forest_model.pkl and scaler.pkl.

📊 Model Performance

The model was evaluated on a dataset encompassing 22 classes (e.g., Mango_Healthy, Guava_Diseased, etc. - 11 plant types × 2 conditions).
Metric	Value
Classification Accuracy	~75%
