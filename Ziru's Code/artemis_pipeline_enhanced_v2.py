#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artemis Pipeline (Enhanced English Version)
==========================================

End-to-end Android malware analysis pipeline that:
1) Unpacks APKs (via apktool)
2) Extracts static & binary features
3) Generates LLM-based semantic descriptions for static features
4) Vectorizes features into tensors
5) Runs a dual-branch 1D CNN classifier (binary branch + text branch)
6) Produces an HTML diagnostic report (optional PDF conversion hook)

This file is designed to be a drop-in, extended replacement for a basic
`apk_processor.py`. It keeps public function names intuitive and adds new ones
to cover the full PRD.

⚠️ Requirements / External Tools
- apktool must be installed on your PATH (https://ibotpeaches.github.io/Apktool/)
- Python deps (see requirements additions below)
- A CNN checkpoint is optional for inference; if not present, the model
  will be created untrained (for demo).

TIP: You can start with HTML reports and later add PDF conversion as needed.

Author: Artemis Team
License: MIT (adjust to your project’s needs)
"""

import os
import re
import io
import sys
import json
import time
import math
import shutil
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# --- Optional / heavy deps: import guarded so this file can be imported without full env ---
try:
    # For static manifest parsing (works on xml produced by apktool)
    from lxml import etree
except Exception:
    etree = None

try:
    # For binary features (DEX)
    from androguard.core.bytecodes.apk import APK
    from androguard.core.bytecodes.dvm import DalvikVMFormat
    from androguard.core.analysis.analysis import Analysis
except Exception:
    APK = DalvikVMFormat = Analysis = None

try:
    # Text embeddings
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    # Vectorization
    import numpy as np
    from sklearn.feature_extraction import FeatureHasher
except Exception:
    np = FeatureHasher = None

try:
    # Keras / TF for CNN
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf = keras = layers = None

try:
    # HTML reporting
    from jinja2 import Template
except Exception:
    Template = None


# -------------------------------
# Data Structures
# -------------------------------

@dataclass
class StaticFeatures:
    permissions: List[str]
    api_calls: List[str]
    urls: List[str]
    uses_features: List[str]
    providers: List[str]
    receivers: List[str]
    services: List[str]
    activities: List[str]
    package_name: Optional[str] = None
    app_label: Optional[str] = None
    version_name: Optional[str] = None
    version_code: Optional[str] = None


@dataclass
class BinaryFeatures:
    instruction_sequences: List[str]  # flattened opcode sequences per method
    method_calls: List[str]


@dataclass
class LLMDescriptions:
    # map "feature-key" -> description string
    by_feature: Dict[str, str]


@dataclass
class ArtemisTensor:
    # Dual-branch tensors (ready for 1D CNN)
    # branch_a: hashed vectors from opcode/API/permissions (shape: [H])
    # branch_b: semantic text embedding vectors from LLM descriptions (shape: [E])
    branch_a: "np.ndarray"
    branch_b: "np.ndarray"


@dataclass
class ClassificationResult:
    is_malicious: bool
    confidence: float
    logits: Optional[float] = None


# -------------------------------
# 1) APK Unpacking (apktool)
# -------------------------------

def unpack_apk(apk_path: str, output_dir: str) -> bool:
    """Unpack an APK to `output_dir` using apktool.

    Returns True if successful, False otherwise.
    """
    if not os.path.exists(apk_path):
        print(f"[ERROR] APK not found: {apk_path}")
        return False

    if os.path.exists(output_dir):
        print(f"[WARN] Output dir exists -> removing: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            print(f"[ERROR] Failed to remove old output dir: {e}")
            return False

    try:
        cmd = ["apktool", "d", "-f", apk_path, "-o", output_dir]
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] apktool decode success")
        return True
    except FileNotFoundError:
        print("[ERROR] apktool not found on PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print("[ERROR] apktool failed")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False


# -------------------------------
# 2) Static Feature Extraction
# -------------------------------

ANDROID_NS = "http://schemas.android.com/apk/res/android"

def _attr(elem, name: str) -> Optional[str]:
    """Read namespaced attribute `android:name` safely."""
    return elem.attrib.get(f"{{{ANDROID_NS}}}{name}") or elem.attrib.get(name)


def extract_static_features(decompiled_apk_dir: str) -> StaticFeatures:
    """Extract manifest-driven static features and quick URL/API scans from smali.

    Works on apktool-decoded directory structure.
    """
    manifest = os.path.join(decompiled_apk_dir, "AndroidManifest.xml")
    if not os.path.exists(manifest):
        print(f"[ERROR] AndroidManifest.xml not found under: {decompiled_apk_dir}")
        # return empty containers
        return StaticFeatures([], [], [], [], [], [], [], [])

    if etree is None:
        print("[WARN] lxml not available, cannot parse manifest; returning minimal features.")
        return StaticFeatures([], [], [], [], [], [], [], [])

    perms = set()
    apis = set()
    urls = set()
    uses_features = set()
    providers, receivers, services, activities = set(), set(), set(), set()

    package_name = app_label = version_name = version_code = None

    try:
        tree = etree.parse(manifest)
        root = tree.getroot()

        package_name = root.attrib.get("package")
        version_code = root.attrib.get("versionCode") or _attr(root, "versionCode")
        version_name = root.attrib.get("versionName") or _attr(root, "versionName")

        # application label (best-effort)
        app_node = root.find("application")
        if app_node is not None:
            app_label = _attr(app_node, "label")

        # permissions
        for node in root.findall("uses-permission"):
            name = _attr(node, "name")
            if name:
                perms.add(name)
        for node in root.findall("permission"):
            name = _attr(node, "name")
            if name:
                perms.add(name)

        # uses-features
        for node in root.findall("uses-feature"):
            name = _attr(node, "name")
            if name:
                uses_features.add(name)

        # components
        for tag, bucket in [
            ("activity", activities),
            ("service", services),
            ("receiver", receivers),
            ("provider", providers),
        ]:
            for node in root.findall(f".//{tag}"):
                name = _attr(node, "name")
                if name:
                    bucket.add(name)

        # lightweight scan in smali to collect API call signatures & URLs
        smali_root = os.path.join(decompiled_apk_dir, "smali")
        if os.path.isdir(smali_root):
            api_pattern = re.compile(r"L[^;]+;->[a-zA-Z0-9_$<>_]+\([^\)]*\)[^;]*;")
            url_pattern = re.compile(
                r"(https?://[a-zA-Z0-9.\-]+(?::[0-9]+)?(?:/[a-zA-Z0-9\-._~:/?#@!$&'()*+,;=%]*)?)"
            )
            for root_dir, _, files in os.walk(smali_root):
                for fname in files:
                    if fname.endswith(".smali"):
                        path = os.path.join(root_dir, fname)
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                apis.update(api_pattern.findall(content))
                                urls.update(url_pattern.findall(content))
                        except Exception:
                            # ignore unreadable files
                            pass
        else:
            print(f"[WARN] smali/ not found under {decompiled_apk_dir}; API/URL scan skipped.")

    except Exception as e:
        print(f"[ERROR] Manifest parse failed: {e}")

    return StaticFeatures(
        permissions=sorted(perms),
        api_calls=sorted(apis),
        urls=sorted(urls),
        uses_features=sorted(uses_features),
        providers=sorted(providers),
        receivers=sorted(receivers),
        services=sorted(services),
        activities=sorted(activities),
        package_name=package_name,
        app_label=app_label,
        version_name=version_name,
        version_code=version_code,
    )


# -------------------------------
# 3) Binary Feature Extraction (DEX/Androguard)
# -------------------------------

def extract_binary_features(apk_path: str) -> BinaryFeatures:
    """Load the APK via Androguard and extract opcode sequences & method-call xrefs."""
    if APK is None:
        print("[WARN] Androguard not available; returning empty binary features.")
        return BinaryFeatures([], [])

    if not os.path.exists(apk_path):
        print(f"[ERROR] APK not found for binary extraction: {apk_path}")
        return BinaryFeatures([], [])

    sequences = []
    calls = set()

    try:
        a = APK(apk_path)
        d = DalvikVMFormat(a.get_dex())
        dx = Analysis(d)
        dx.create_xref()

        for method in d.get_methods():
            # skip external lib methods in sequences, but still collect calls via xrefs
            for _, to_m, _ in dx.get_xref_from(method):
                calls.add(str(to_m))

            code = method.get_code()
            if code is None:
                continue
            insns = []
            for insn in method.get_instructions():
                insns.append(insn.get_name())
            if insns:
                sequences.append(" ".join(insns))

    except Exception as e:
        print(f"[ERROR] Binary feature extraction failed: {e}")

    return BinaryFeatures(instruction_sequences=sequences, method_calls=sorted(list(calls)))


# -------------------------------
# 4) LLM-based Descriptions
# -------------------------------

class LLMDescriber:
    """Generate short, security-focused descriptions for static features.
    Uses local embeddings if LLM API is unavailable; prompts kept minimal.
    """

    DEFAULT_PROMPT = (
        "You are a security analyst. Given an Android feature name, write one sentence "
        "explaining why it might be risky or noteworthy in malware analysis. Be concise."
    )

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or "").lower()
        self.model = model

        # Lazy imports for providers
        self._openai_client = None
        if self.provider == "openai":
            try:
                import openai  # type: ignore
                self._openai_client = openai.OpenAI()
            except Exception:
                self._openai_client = None

    @staticmethod
    def _hash_key(kind: str, value: str) -> str:
        hv = hashlib.sha1(f"{kind}:{value}".encode("utf-8")).hexdigest()[:16]
        return f"{kind}:{hv}"

    def _describe_openai(self, feature_list: List[str]) -> Dict[str, str]:
        out = {}
        if self._openai_client is None:
            return out
        sys_prompt = self.DEFAULT_PROMPT
        for feat in feature_list:
            try:
                prompt = f"{sys_prompt}\nFeature: {feat}"
                resp = self._openai_client.chat.completions.create(
                    model=self.model or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You write short, factual analyses."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=64,
                )
                text = (resp.choices[0].message.content or "").strip()
                out[self._hash_key("feat", feat)] = text
            except Exception:
                # Skip on failure; can be filled later offline
                pass
        return out

    def describe(self, static: StaticFeatures) -> LLMDescriptions:
        # Concatenate all relevant feature strings for description
        feature_pool = (
            static.permissions
            + static.api_calls
            + static.urls
            + static.uses_features
        )

        # Try OpenAI if requested; otherwise fall back to heuristic descriptions
        by_feature: Dict[str, str] = {}
        if self.provider == "openai":
            by_feature.update(self._describe_openai(feature_pool))

        # Fill missing with heuristics
        for feat in feature_pool:
            key = self._hash_key("feat", feat)
            if key not in by_feature:
                by_feature[key] = self._fallback_desc(feat)

        return LLMDescriptions(by_feature=by_feature)

    @staticmethod
    def _fallback_desc(feat: str) -> str:
        # Heuristic one-liners without external API calls
        if "READ_SMS" in feat or "SEND_SMS" in feat:
            return "May allow reading or sending text messages, often abused for fraud or exfiltration."
        if "RECEIVE_BOOT_COMPLETED" in feat:
            return "Allows code to run at startup, common in persistence mechanisms."
        if "BIND_ACCESSIBILITY_SERVICE" in feat:
            return "Grants high-privilege accessibility control, risky for keylogging or UI hijacking."
        if feat.startswith("http"):
            return "Network endpoint observed in code; review for data exfiltration or C2."
        if "getDeviceId" in feat or "READ_PHONE_STATE" in feat:
            return "May access device identifiers, potentially used for tracking or fingerprinting."
        if "INTERNET" in feat:
            return "Allows network communication; benign but necessary for data exfiltration."
        return "Potentially relevant artifact; requires contextual review for risk assessment."


# -------------------------------
# 5) Tensorization (Hashing + Embeddings) 
# -------------------------------

class TensorBuilder:
    """Create fixed-size vectors for a dual-branch 1D CNN.
    - Branch A: feature hashing from permissions/APIs/urls/opcodes/method calls
    - Branch B: text embedding that compresses LLM descriptions
    """

    def __init__(self, hash_dim: int = 2048, embed_model: str = "all-MiniLM-L6-v2"):
        self.hash_dim = hash_dim
        self.embed_model_name = embed_model
        self._embedder = None
        if SentenceTransformer is not None:
            try:
                self._embedder = SentenceTransformer(self.embed_model_name)
            except Exception:
                self._embedder = None

    @staticmethod
    def _flatten_strings(*bags: List[str]) -> List[str]:
        out: List[str] = []
        for bag in bags:
            out.extend(bag)
        return out

    def _hash_bag(self, strings: List[str]) -> "np.ndarray":
        if FeatureHasher is None or np is None:
            raise ImportError("scikit-learn and numpy are required for hashing")
        hasher = FeatureHasher(n_features=self.hash_dim, input_type="string", alternate_sign=False)
        # FeatureHasher expects an iterator of raw strings; we wrap as a list-of-list
        X = hasher.transform([strings])
        vec = X.toarray()[0].astype("float32")
        return vec

    def _embed_text(self, texts: List[str]) -> "np.ndarray":
        if np is None:
            raise ImportError("numpy is required")
        if not texts:
            return np.zeros((384,), dtype="float32")  # typical dim for MiniLM
        if self._embedder is None:
            # Fallback: simple hashed bag-of-words average (deterministic)
            hasher = FeatureHasher(n_features=384, input_type="string", alternate_sign=False)
            X = hasher.transform([texts])
            return X.toarray()[0].astype("float32")
        emb = self._embedder.encode(texts, normalize_embeddings=True)
        if isinstance(emb, list):
            emb = np.array(emb)
        # aggregate to a single embedding (mean-pool)
        vec = emb.mean(axis=0).astype("float32")
        return vec

    def build(self, static: StaticFeatures, binary: BinaryFeatures, llm: LLMDescriptions) -> ArtemisTensor:
        # Branch A: hashed signals from static + binary artifacts
        bag_a = self._flatten_strings(
            static.permissions,
            static.api_calls,
            static.urls,
            static.uses_features,
            binary.instruction_sequences,
            binary.method_calls,
        )
        branch_a_vec = self._hash_bag(bag_a)  # shape: [H]

        # Branch B: pooled text embedding of all LLM descriptions
        all_desc = list(llm.by_feature.values())
        branch_b_vec = self._embed_text(all_desc)  # shape: [E]

        return ArtemisTensor(branch_a=branch_a_vec, branch_b=branch_b_vec)


# -------------------------------
# 6) Dual-Branch 1D CNN
# -------------------------------

class DualBranchCNN:
    """Two 1D CNN branches (hash-vector & embedding) merged for binary classification."""
    def __init__(self, hash_dim: int, embed_dim: int, lr: float = 1e-3):
        if tf is None or keras is None or layers is None:
            raise ImportError("TensorFlow/Keras is required to build the CNN model.")
        self.hash_dim = hash_dim
        self.embed_dim = embed_dim
        self.model = self._build_model(hash_dim, embed_dim, lr)

    @staticmethod
    def _conv_block(x, filters: int, k: int = 5):
        x = layers.Conv1D(filters, k, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(pool_size=2)(x)
        return x

    def _build_model(self, hash_dim: int, embed_dim: int, lr: float):
        # Branch A (hash)
        in_a = keras.Input(shape=(hash_dim, 1), name="hash_branch")
        xa = self._conv_block(in_a, 32)
        xa = self._conv_block(xa, 64)
        xa = layers.Flatten()(xa)

        # Branch B (embedding)
        in_b = keras.Input(shape=(embed_dim, 1), name="embed_branch")
        xb = self._conv_block(in_b, 16)
        xb = self._conv_block(xb, 32)
        xb = layers.Flatten()(xb)

        # Merge
        x = layers.Concatenate()([xa, xb])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=[in_a, in_b], outputs=out, name="DualBranchCNN")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def predict_one(self, tensor: ArtemisTensor) -> ClassificationResult:
        # reshape to (1, L, 1) for 1D CNN
        a = tensor.branch_a.reshape(1, -1, 1)
        b = tensor.branch_b.reshape(1, -1, 1)
        prob = float(self.model.predict([a, b], verbose=0)[0][0])
        return ClassificationResult(is_malicious=(prob >= 0.5), confidence=prob, logits=math.log(prob + 1e-9))  # logit-like

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)

    def save_weights(self, path: str) -> None:
        self.model.save_weights(path)


# -------------------------------
# 7) Report Generation (HTML)
# -------------------------------

_REPORT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Artemis Diagnostic Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; }
    .bad { background: #fee2e2; color: #991b1b; }
    .good { background: #dcfce7; color: #166534; }
    .muted { color: #64748b; }
    h1 { margin-bottom: 0; }
    h2 { margin-top: 32px; }
    code { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }
    ul { line-height: 1.6; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; }
    .small { font-size: 13px; }
  </style>
</head>
<body>
  <h1>Artemis Diagnostic Report</h1>
  <p class="muted small">Generated at {{ generated_at }}</p>

  <p>
    <span class="badge {{ 'bad' if result.is_malicious else 'good' }}">
      {{ 'MALICIOUS' if result.is_malicious else 'BENIGN' }}
    </span>
    &nbsp;Confidence: <b>{{ '{:.2%}'.format(result.confidence) }}</b>
  </p>

  <h2>App Metadata</h2>
  <div class="grid">
    <div class="card small">
      <b>Package</b><br/>{{ static.package_name or '—' }}
    </div>
    <div class="card small">
      <b>App Label</b><br/>{{ static.app_label or '—' }}
    </div>
    <div class="card small">
      <b>Version Name</b><br/>{{ static.version_name or '—' }}
    </div>
    <div class="card small">
      <b>Version Code</b><br/>{{ static.version_code or '—' }}
    </div>
  </div>

  <h2>Suspicious Artifacts (Heuristics)</h2>
  <ul>
    {% for s in suspicious %}
      <li><code>{{ s }}</code></li>
    {% else %}
      <li class="muted">No obvious high-risk artifacts detected by heuristics.</li>
    {% endfor %}
  </ul>

  <h2>LLM Explanations (Top-10)</h2>
  <ol>
    {% for k, v in llm_items[:10] %}
      <li><code>{{ k }}</code> — {{ v }}</li>
    {% else %}
      <li class="muted">No LLM descriptions available.</li>
    {% endfor %}
  </ol>

  <h2>Raw Feature Counts</h2>
  <ul class="small">
    <li>Permissions: {{ static.permissions|length }}</li>
    <li>API Calls: {{ static.api_calls|length }}</li>
    <li>URLs: {{ static.urls|length }}</li>
    <li>Uses-Features: {{ static.uses_features|length }}</li>
    <li>Activities: {{ static.activities|length }}</li>
    <li>Services: {{ static.services|length }}</li>
    <li>Receivers: {{ static.receivers|length }}</li>
    <li>Providers: {{ static.providers|length }}</li>
    <li>Instruction Sequences: {{ binary.instruction_sequences|length }}</li>
    <li>Method Calls: {{ binary.method_calls|length }}</li>
  </ul>

  <hr/>
  <p class="small muted">
    This report combines static analysis, heuristic signals, LLM-based explanations, and a dual-branch CNN prediction.
    Always validate results with additional dynamic analysis when possible.
  </p>
</body>
</html>
"""

def _heuristic_suspicious(static: StaticFeatures) -> List[str]:
    """Simple allowlist to surface commonly risky items in the report."""
    HOT = [
        "android.permission.READ_SMS",
        "android.permission.SEND_SMS",
        "android.permission.RECEIVE_SMS",
        "android.permission.RECEIVE_BOOT_COMPLETED",
        "android.permission.BIND_ACCESSIBILITY_SERVICE",
        "android.permission.READ_CONTACTS",
        "android.permission.CALL_PHONE",
        "android.permission.WRITE_SETTINGS",
        "android.permission.SYSTEM_ALERT_WINDOW",
    ]
    out = []
    for p in static.permissions:
        if p in HOT:
            out.append(p)
    for u in static.urls:
        if any(host in u for host in (".onion", "pastebin", "bit.ly", "tinyurl", "dropbox.com")):
            out.append(u)
    return out


def render_report_html(
    output_path: str,
    result: ClassificationResult,
    static: StaticFeatures,
    binary: BinaryFeatures,
    llm: LLMDescriptions,
    generated_at: Optional[str] = None,
) -> str:
    """Render and write an HTML report. Returns the saved path.
    If Jinja2 is not installed, falls back to a dependency-free HTML generator.
    """
    def _escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
        )

    if Template is not None:
        t = Template(_REPORT_HTML)
        html = t.render(
            result=result,
            static=static,
            binary=binary,
            llm_items=list(llm.by_feature.items()),
            suspicious=_heuristic_suspicious(static),
            generated_at=generated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        # Simple fallback without Jinja2
        gen_at = generated_at or time.strftime("%Y-%m-%d %H:%M:%S")
        verdict = "MALICIOUS" if result.is_malicious else "BENIGN"
        badge_class = "bad" if result.is_malicious else "good"

        def _li(items):
            return "\n".join(f"<li><code>{_escape(str(x))}</code></li>" for x in items)

        llm_items = list(llm.by_feature.items())[:10]
        llm_list = "\n".join(
            f"<li><code>{_escape(k)}</code> — {_escape(v)}</li>" for k, v in llm_items
        )
        suspicious = _heuristic_suspicious(static)
        susp_html = _li(suspicious) if suspicious else '<li class="muted">No obvious high-risk artifacts detected by heuristics.</li>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Artemis Diagnostic Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; }}
    .bad {{ background: #fee2e2; color: #991b1b; }}
    .good {{ background: #dcfce7; color: #166534; }}
    .muted {{ color: #64748b; }}
    h1 {{ margin-bottom: 0; }}
    h2 {{ margin-top: 32px; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }}
    ul {{ line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; }}
    .small {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Artemis Diagnostic Report</h1>
  <p class="muted small">Generated at {gen_at}</p>

  <p>
    <span class="badge {badge_class}">{verdict}</span>
    &nbsp;Confidence: <b>{result.confidence:.2%}</b>
  </p>

  <h2>App Metadata</h2>
  <div class="grid">
    <div class="card small"><b>Package</b><br/>{_escape(str(static.package_name or '—'))}</div>
    <div class="card small"><b>App Label</b><br/>{_escape(str(static.app_label or '—'))}</div>
    <div class="card small"><b>Version Name</b><br/>{_escape(str(static.version_name or '—'))}</div>
    <div class="card small"><b>Version Code</b><br/>{_escape(str(static.version_code or '—'))}</div>
  </div>

  <h2>Suspicious Artifacts (Heuristics)</h2>
  <ul>
    {susp_html}
  </ul>

  <h2>LLM Explanations (Top-10)</h2>
  <ol>
    {llm_list if llm_items else '<li class="muted">No LLM descriptions available.</li>'}
  </ol>

  <h2>Raw Feature Counts</h2>
  <ul class="small">
    <li>Permissions: {len(static.permissions)}</li>
    <li>API Calls: {len(static.api_calls)}</li>
    <li>URLs: {len(static.urls)}</li>
    <li>Uses-Features: {len(static.uses_features)}</li>
    <li>Activities: {len(static.activities)}</li>
    <li>Services: {len(static.services)}</li>
    <li>Receivers: {len(static.receivers)}</li>
    <li>Providers: {len(static.providers)}</li>
    <li>Instruction Sequences: {len(binary.instruction_sequences)}</li>
    <li>Method Calls: {len(binary.method_calls)}</li>
  </ul>

  <hr/>
  <p class="small muted">
    This report combines static analysis, heuristic signals, LLM-based explanations, and a dual-branch CNN prediction.
    Always validate results with additional dynamic analysis when possible.
  </p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# -------------------------------
# 8) Orchestration Helpers
# -------------------------------

def analyze_apk(
    apk_path: str,
    work_dir: str,
    cnn_weights_path: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    report_output: Optional[str] = None,
) -> Dict[str, object]:
    """Full pipeline for a single APK -> returns a JSON-able dict with results.

    - Creates/cleans a working directory under `work_dir`
    - Unpacks via apktool
    - Extracts static + binary features
    - Generates LLM descriptions
    - Builds tensors
    - Loads CNN (if weights provided) and predicts; otherwise returns a neutral score
    - Writes HTML report if path provided

    Returns a dictionary you can dump to JSON, and optionally writes a report.
    """
    os.makedirs(work_dir, exist_ok=True)
    decompiled_dir = os.path.join(work_dir, "apk_decoded")
    if not unpack_apk(apk_path, decompiled_dir):
        raise RuntimeError("apktool unpack failed")

    static = extract_static_features(decompiled_dir)
    binary = extract_binary_features(apk_path)

    describer = LLMDescriber(provider=llm_provider, model=llm_model)
    llm = describer.describe(static)

    tb = TensorBuilder()
    tensor = tb.build(static, binary, llm)

    # Create / load CNN
    embed_dim = tensor.branch_b.shape[0]
    model = DualBranchCNN(hash_dim=tensor.branch_a.shape[0], embed_dim=embed_dim)
    if cnn_weights_path and os.path.exists(cnn_weights_path):
        model.load_weights(cnn_weights_path)
    else:
        print("[WARN] No CNN weights provided; using randomly initialized model (prediction is not meaningful).")

    result = model.predict_one(tensor)

    # Render report if requested
    saved_report = None
    if report_output:
        saved_report = render_report_html(
            output_path=report_output,
            result=result,
            static=static,
            binary=binary,
            llm=llm,
        )

    return {
        "apk_path": apk_path,
        "static": asdict(static),
        "binary": asdict(binary),
        "llm_descriptions": llm.by_feature,
        "classification": asdict(result),
        "report_path": saved_report,
    }


# -------------------------------
# 9) CLI
# -------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Artemis APK Analyzer (Enhanced)")
    p.add_argument("apk", help="Path to APK file")
    p.add_argument("--work", default="artemis_workdir", help="Working directory for decode & temp")
    p.add_argument("--weights", default=None, help="Path to CNN weights (.weights.h5 or .keras)")
    p.add_argument("--llm-provider", default=None, choices=[None, "openai", "hf"], help="LLM provider (optional)")
    p.add_argument("--llm-model", default=None, help="LLM model name for provider (optional)")
    p.add_argument("--report", default="artemis_report.html", help="Output HTML report path")
    args = p.parse_args()

    out = analyze_apk(
        apk_path=args.apk,
        work_dir=args.work,
        cnn_weights_path=args.weights,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        report_output=args.report,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
