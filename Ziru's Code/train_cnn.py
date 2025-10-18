# train_cnn.py
"""
Train DualBranchCNN on a dataset CSV with columns:
apk_path,label
/path/to/app1.apk,1
/path/to/app2.apk,0
...
This script extracts features, builds tensors, trains, evaluates, and saves weights.
"""
import os, csv, json, time
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .extractor import unpack_apk, extract_static_features, extract_binary_features
from .llm import LLMDescriber
from .tensorizer import TensorBuilder
from .model import DualBranchCNN

def load_dataset(csv_path: str) -> List[Tuple[str,int]]:
    data=[]
    with open(csv_path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            apk=row["apk_path"].strip()
            y=int(row["label"])
            data.append((apk,y))
    return data

def build_tensor_for_apk(apk_path: str, cache_dir: str, llm_provider=None, llm_model=None):
    os.makedirs(cache_dir, exist_ok=True)
    npz = os.path.join(cache_dir, "tensor.npz")
    if os.path.exists(npz):
        arr = np.load(npz)
        return arr["a"], arr["b"]

    decoded = os.path.join(cache_dir, "decoded")
    if not unpack_apk(apk_path, decoded):
        raise RuntimeError(f"apktool failed for {apk_path}")
    static = extract_static_features(decoded)
    binary = extract_binary_features(apk_path)
    llm = LLMDescriber(provider=llm_provider, model=llm_model).describe(static)
    tb = TensorBuilder()
    tensor = tb.build(static, binary, llm)
    a, b = tensor.branch_a, tensor.branch_b
    np.savez(npz, a=a, b=b)
    return a, b

def train(csv_path: str, work_root: str, out_weights: str, epochs: int=5, batch: int=8,
          llm_provider=None, llm_model=None):
    items = load_dataset(csv_path)
    A=[]; B=[]; Y=[]
    for i,(apk,y) in enumerate(items, start=1):
        cache = os.path.join(work_root, f"apk_{i}")
        try:
            a,b = build_tensor_for_apk(apk, cache, llm_provider, llm_model)
            A.append(a); B.append(b); Y.append(y)
        except Exception as e:
            print(f"[WARN] skip {apk}: {e}")

    A=np.array(A, dtype="float32"); B=np.array(B, dtype="float32"); Y=np.array(Y, dtype="float32")
    if len(Y)==0:
        raise RuntimeError("No valid samples to train. Check dataset paths.")

    # derive shapes
    hash_dim=A.shape[1]; embed_dim=B.shape[1]
    # reshape for 1D CNN
    A_ = A.reshape((-1, hash_dim, 1)); B_ = B.reshape((-1, embed_dim, 1))

    # split
    Xtr_a, Xte_a, Xtr_b, Xte_b, ytr, yte = train_test_split(A_, B_, Y, test_size=0.25, random_state=42, stratify=Y if len(np.unique(Y))>1 else None)

    model = DualBranchCNN(hash_dim=hash_dim, embed_dim=embed_dim)
    model.fit(Xtr_a, Xtr_b, ytr, epochs=epochs, batch_size=batch, val_data=(Xte_a, Xte_b, yte))
    model.save_weights(out_weights)

    # evaluate
    probs = model.model.predict([Xte_a, Xte_b], verbose=0).reshape(-1)
    preds = (probs>=0.5).astype("int32")
    acc = float(accuracy_score(yte, preds))
    prec = float(precision_score(yte, preds, zero_division=0))
    rec = float(recall_score(yte, preds, zero_division=0))
    f1 = float(f1_score(yte, preds, zero_division=0))

    metrics = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"n_val":int(len(yte))}
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(work_root,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2)

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train DualBranchCNN on APK dataset")
    p.add_argument("--dataset", required=True, help="CSV with columns apk_path,label")
    p.add_argument("--work", default="train_work", help="Working directory for decode/cache")
    p.add_argument("--out", default="cnn_model.weights.h5", help="Output weights path")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--llm-provider", default=None, choices=[None,"openai","hf"])
    p.add_argument("--llm-model", default=None)
    args = p.parse_args()
    train(args.dataset, args.work, args.out, args.epochs, args.batch, args.llm_provider, args.llm_model)
