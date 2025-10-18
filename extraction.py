import os
import numpy as np
from androguard.misc import AnalyzeAPK
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

# Read permissions list
with open("unique_permissions.txt", "r", encoding="utf-8") as perm_file:
    permArr = [line.strip() for line in perm_file if line.strip()]  # Remove empty lines

# Read API calls list
with open("unique_api_calls_minimized.txt", "r", encoding="utf-8") as api_file:
    apiArr = [line.strip() for line in api_file if line.strip()]  # Remove empty lines

# Combine features and create feature index dictionary
featArr = permArr + apiArr
featIdx = {feat: idx for idx, feat in enumerate(featArr)}
input_size = len(featArr)

# Feature vector generator
def apkVector(apk_path):
    vec = np.zeros(input_size, dtype=np.float32)
    try:
        a, d, dx = AnalyzeAPK(apk_path)
        for perm in a.get_permissions():
            if perm in featIdx:
                vec[featIdx[perm]] = 1.0
        for method in dx.get_methods():
            for _, call, _ in method.get_xref_to():
                classname = call.class_name[1:-1].replace("/", ".")
                methodname = call.name
                if classname.startswith("android.") or classname.startswith("java."):
                    token = f"{classname}.{methodname}"
                    if token in featIdx:
                        vec[featIdx[token]] = 1.0
    except Exception as e:
        print(f"Error processing {apk_path}: {e}")
    return vec

# Dataset loading (benign + malicious samples)
X, y = [], []
root_dir = "dataset"  # top-level dataset folder

for label, subdir in [(1, "malicious"), (0, "benign")]:
    apk_folder = os.path.join(root_dir, subdir)
    for root, _, files in os.walk(apk_folder):
        for apk in files:
            if apk.endswith(".apk"):
                X.append(apkVector(os.path.join(root, apk)))
                y.append(label)

# Convert to numpy and reshape for CNN
X = np.array(X)
y = np.array(y)
X = np.expand_dims(X, -1)  # add channel dim

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define CNN model, confused on how to train properly, so this is a model suggested by AI
model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(input_size, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Basic compilation, use binary crossentropy for binary classification and adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train and validate
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, shuffle=True)

# Evaluate model performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the trained model
model.save("apk_malware_cnn_model.keras")

# To load the model later: 
# from tensorflow.keras.models import load_model
# model = load_model("malware_detector_cnn.h5")