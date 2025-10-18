# extractor.py
import os, re, shutil, subprocess
from typing import Optional, List, Set
from .types import StaticFeatures, BinaryFeatures

ANDROID_NS = "http://schemas.android.com/apk/res/android"

def unpack_apk(apk_path: str, output_dir: str) -> bool:
    if not os.path.exists(apk_path):
        print(f"[ERROR] APK not found: {apk_path}")
        return False
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            print(f"[ERROR] remove output_dir failed: {e}")
            return False
    try:
        cmd = ["apktool", "d", "-f", apk_path, "-o", output_dir]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("[ERROR] apktool not found")
        return False
    except subprocess.CalledProcessError as e:
        print("[ERROR] apktool failed"); print(e.stdout); print(e.stderr)
        return False

def _attr(elem, name: str) -> Optional[str]:
    return elem.attrib.get(f"{{{ANDROID_NS}}}{name}") or elem.attrib.get(name)

def extract_static_features(decompiled_apk_dir: str) -> StaticFeatures:
    manifest = os.path.join(decompiled_apk_dir, "AndroidManifest.xml")
    if not os.path.exists(manifest):
        print(f"[ERROR] Manifest missing under {decompiled_apk_dir}")
        return StaticFeatures([],[],[],[],[],[],[],[])

    try:
        from lxml import etree
    except Exception:
        print("[WARN] lxml missing; returning empty static features")
        return StaticFeatures([],[],[],[],[],[],[],[])

    perms:Set[str]=set(); apis:Set[str]=set(); urls:Set[str]=set()
    uses_features:Set[str]=set()
    providers, receivers, services, activities = set(), set(), set(), set()
    package_name=app_label=version_name=version_code=None

    try:
        tree = etree.parse(manifest); root = tree.getroot()
        package_name = root.attrib.get("package")
        version_code = root.attrib.get("versionCode") or _attr(root,"versionCode")
        version_name = root.attrib.get("versionName") or _attr(root,"versionName")
        app_node = root.find("application")
        if app_node is not None: app_label = _attr(app_node,"label")

        for node in root.findall("uses-permission"):
            n = _attr(node,"name"); 
            if n: perms.add(n)
        for node in root.findall("permission"):
            n = _attr(node,"name"); 
            if n: perms.add(n)

        for node in root.findall("uses-feature"):
            n = _attr(node,"name"); 
            if n: uses_features.add(n)

        for tag,bucket in [("activity",activities),("service",services),("receiver",receivers),("provider",providers)]:
            for node in root.findall(f".//{tag}"):
                n = _attr(node,"name"); 
                if n: bucket.add(n)

        smali_root = os.path.join(decompiled_apk_dir,"smali")
        if os.path.isdir(smali_root):
            api_pattern = re.compile(r"L[^;]+;->[a-zA-Z0-9_$<>_]+\([^\)]*\)[^;]*;")
            url_pattern = re.compile(r"(https?://[a-zA-Z0-9.\-]+(?::[0-9]+)?(?:/[a-zA-Z0-9\-._~:/?#@!$&'()*+,;=%]*)?)")
            for root_dir,_,files in os.walk(smali_root):
                for fname in files:
                    if not fname.endswith(".smali"): continue
                    path = os.path.join(root_dir,fname)
                    try:
                        with open(path,"r",encoding="utf-8",errors="ignore") as f:
                            s=f.read()
                            apis.update(api_pattern.findall(s))
                            urls.update(url_pattern.findall(s))
                    except Exception: pass
        else:
            print(f"[WARN] smali/ not found: {smali_root}")
    except Exception as e:
        print(f"[ERROR] manifest parse failed: {e}")

    return StaticFeatures(
        permissions=sorted(perms), api_calls=sorted(apis), urls=sorted(urls),
        uses_features=sorted(uses_features), providers=sorted(providers),
        receivers=sorted(receivers), services=sorted(services),
        activities=sorted(activities), package_name=package_name,
        app_label=app_label, version_name=version_name, version_code=version_code
    )

def extract_binary_features(apk_path: str) -> BinaryFeatures:
    try:
        from androguard.core.bytecodes.apk import APK
        from androguard.core.bytecodes.dvm import DalvikVMFormat
        from androguard.core.analysis.analysis import Analysis
    except Exception:
        print("[WARN] androguard missing; returning empty binary features")
        return BinaryFeatures([],[])
    import os
    if not os.path.exists(apk_path):
        print(f"[ERROR] APK not found: {apk_path}")
        return BinaryFeatures([],[])

    sequences=[]; calls=set()
    try:
        a=APK(apk_path); d=DalvikVMFormat(a.get_dex()); dx=Analysis(d); dx.create_xref()
        for m in d.get_methods():
            for _,to_m,_ in dx.get_xref_from(m):
                calls.add(str(to_m))
            code = m.get_code()
            if code is None: continue
            ins=[insn.get_name() for insn in m.get_instructions()]
            if ins: sequences.append(" ".join(ins))
    except Exception as e:
        print(f"[ERROR] binary feature extraction failed: {e}")
    return BinaryFeatures(sequences, sorted(list(calls)))
