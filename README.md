🌿 Leaf Vein Classification System using Machine Learning

This project classifies plant leaves into their species and health condition using vein structure analysis.
It extracts the skeleton pattern, converts it into numerical features, and uses a Random Forest classifier to predict leaf type.

This system works on any uploaded leaf image and provides:
✔️ Skeletonized vein map
✔️ Extracted numerical features
✔️ Machine learning prediction
✔️ Classification accuracy
✔️ A Streamlit-based testing UI

📁 Dataset Structure

Your dataset must follow this format:

Leaf_Dataset/
   ├── Class_1/
   │      ├── img1.jpg
   │      └── img2.jpg
   ├── Class_2/
   │      ├── img1.jpg
   │      └── img2.jpg
   └── Class_n/


Example:

Plants/train/
   ├── Mango_Healthy/
   ├── Mango_Diseased/
   ├── Guava_Healthy/
   ├── Guava_Diseased/
   └── ...

🔬 Methodology
🧮 1. Preprocessing (OpenCV)

Resize and normalize image

Convert to grayscale

Apply Gaussian filter

Binary thresholding

Morphological thinning to obtain vein skeleton

🧪 2. Feature Extraction

From the skeleton we calculate:

Number of vein segments

Length of veins

Mean branch thickness

Area covered by veins

Vein density metrics

Statistical descriptors

Output: a 13-dimensional feature vector.

🌲 3. Model Used: RandomForestClassifier

Works well on structured features

Handles noise and outliers

Robust for biological image patterns

Training:

Train-test split = 80 / 20

Normalization using StandardScaler

Saved as random_forest_model.pkl

📊 Model Performance

Accuracy: ~75%

Evaluated on 22 classes (11 plants × healthy/diseased)

Metrics generated:

Precision

Recall

F1-score

Confusion matrix

🚀 Installation
git clone https://github.com/boyasaikiran/leaf_vein_classification.git
cd leaf_vein_classification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

🧠 Training the Model

Modify dataset path in train_model.py and run:

python train_model.py


This generates:

models/random_forest_model.pkl

models/scaler.pkl

🧪 Testing with Streamlit UI
streamlit run app.py


Streamlit Output Provides:
👍 Original Image
👍 Skeleton Vein Map
👍 Feature Extraction Table
👍 Predicted Leaf Class

🧵 Project Folder Structure
leaf_vein_classification/
   ├── src/
   │     ├── preprocessing.py
   │     └── feature_extraction.py
   ├── models/
   │     ├── random_forest_model.pkl
   │     └── scaler.pkl
   ├── train_model.py
   ├── app.py
   ├── requirements.txt
   └── README.md

🎥 Demo Proof (Video)

You can add here:

📌 Uploaded demonstration video link (Google Drive / YouTube).

👨‍🏫 How to Explain to Guide (Summary)

We extract skeleton veins because veins uniquely identify leaf type.

We convert skeleton into 13 measured features.

We train RandomForest for classification.

Accuracy achieved: ~75%.

Frontend built with Streamlit for live testing.

Works on any leaf uploaded by the user.

📝 Results / Outputs
