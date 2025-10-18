import os
import numpy as np
from tensorflow.keras.models import load_model
from androguard.misc import AnalyzeAPK

# Load feature lists (must be same as training)
with open("unique_permissions.txt", "r", encoding="utf-8") as perm_file:
    permArr = [line.strip() for line in perm_file if line.strip()]
with open("unique_api_calls_minimized.txt", "r", encoding="utf-8") as api_file:
    apiArr = [line.strip() for line in api_file if line.strip()]
featArr = permArr + apiArr
featIdx = {feat: idx for idx, feat in enumerate(featArr)}
input_size = len(featArr)

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
        return None
    return vec

model = load_model("apk_malware_cnn_model.keras")
apk_folder = r"C:\Senior_Proj\cnnTestFolder"

results = []

for apk_file in os.listdir(apk_folder):
    if apk_file.endswith(".apk"):
        apk_path = os.path.join(apk_folder, apk_file)
        feature_vector = apkVector(apk_path)
        if feature_vector is None:
            results.append([apk_file, "Error", "N/A"])
            continue
        feature_vector = np.expand_dims(feature_vector, axis=(0, 2))
        prediction = model.predict(feature_vector)
        malware_prob = float(prediction[0][0])
        label = "MALICIOUS" if malware_prob > 0.5 else "BENIGN"
        confidence = malware_prob if malware_prob > 0.5 else 1 - malware_prob
        results.append([apk_file, label, f"{confidence:.4f}"])

# Print table header
print("{:<40} {:<12} {:<10}".format("APK File", "Classification", "Confidence"))
print("-" * 64)

# Print table rows
for apk_file, label, conf in results:
    print("{:<40} {:<12} {:<10}".format(apk_file, label, conf))