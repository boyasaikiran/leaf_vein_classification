import os
import sys
import joblib
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src paths
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_features

# Paths
DATASET_PATH = r"D:\Plants_2\train"
MODEL_SAVE_PATH = os.path.join("models")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("📂 Loading dataset...")

# Detect class subfolders
subdirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

images, labels, classes = [], [], []

if len(subdirs) == 0:
    print("⚠️ No class subfolders found — treating all images as one class 'Leaves'")
    classes = ["Leaves"]
    for file_name in os.listdir(DATASET_PATH):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(DATASET_PATH, file_name))
            labels.append(0)
else:
    classes = subdirs
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(DATASET_PATH, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(class_dir, file_name))
                labels.append(label)

print(f"✅ Found {len(classes)} class(es): {classes}")
print(f"✅ Loaded {len(images)} images total.\n")

# -------------------------------------------------
# 1️⃣ Preprocess + Extract Features
# -------------------------------------------------
X, y = [], []
for img_path, label in tqdm(zip(images, labels), total=len(images), desc="🔹 Extracting features"):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Unreadable: {img_path}, skipping")
            continue

        if len(img.shape) == 2:  # grayscale → BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        processed, binary, skeleton = preprocess_image(img_path)
        features = extract_features(processed, binary, skeleton)

        if features is not None and len(features) > 0:
            X.append(features)
            y.append(label)
        else:
            print(f"⚠️ Empty features for {img_path}, skipping")

    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("❌ No valid images processed. Check dataset path or preprocessing function.")

print(f"\n✅ Successfully extracted features from {len(X)} images.")
print(f"✅ Feature shape: {X.shape}")

# -------------------------------------------------
# 2️⃣ Split dataset
# -------------------------------------------------
if len(np.unique(y)) < 2:
    print("⚠️ Only one class detected — skipping train/test split.")
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------------------------
# 3️⃣ Scale features
# -------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, "scaler.pkl"))
print("✅ Features normalized and scaler saved.")

# -------------------------------------------------
# 4️⃣ Train RandomForest model
# -------------------------------------------------
print("\n🌲 Training RandomForest model...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("✅ RandomForest training complete!")

# -------------------------------------------------
# 5️⃣ Evaluate model
# -------------------------------------------------
if len(np.unique(y)) > 1:
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
else:
    print("⚠️ Only one class — skipping evaluation metrics.")

# -------------------------------------------------
# 6️⃣ Save trained model
# -------------------------------------------------
model_path = os.path.join(MODEL_SAVE_PATH, "random_forest_model.pkl")
joblib.dump(rf, model_path)
print(f"\n💾 Model saved at: {model_path}")
print("\n✅ Training completed successfully!")
