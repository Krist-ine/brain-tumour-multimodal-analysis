import os
import numpy as np
import cv2
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# ---------------------------------------------------
# LOAD IMAGES AS FLATTENED ARRAYS
# ---------------------------------------------------
def load_images_as_array(root_dir):
    class_names = sorted(os.listdir(root_dir))
    X = []
    y = []

    for label in class_names:
        class_folder = os.path.join(root_dir, label)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))   # Resize to smaller for XGBoost
            img = img.flatten()                # Flatten to 1D vector
            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y, class_names


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
X_train, y_train, class_names = load_images_as_array("data/Training")
X_test, y_test, _ = load_images_as_array("data/Testing")

# Encode labels to integers for XGBoost
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# ---------------------------------------------------
# TRAIN XGBOOST MODEL
# ---------------------------------------------------
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(class_names),
    eval_metric='mlogloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICT & EVALUATE
# ---------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}\n")

# Detailed performance report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Save report
report_df.to_csv("xgboost_classification_report.csv", index=True)
print("\nSaved: xgboost_classification_report.csv")

# Save model
model.save_model("xgboost_model.json")
print("Saved model: xgboost_model.json")
