import os
import re
import subprocess
import zipfile
import io
import shutil
import hashlib
import json
from lxml import etree
from androguard.core.bytecodes.apk import APK
from androguard.core.analysis.analysis import Analysis
from androguard.misc import a
from data_structures import StaticFeatures, BinaryFeatures

def unpack_apk(apk_path: str, output_dir: str) -> bool:
    """Unpacks an APK file using apktool.

    Args:
        apk_path: Path to the APK file.
        output_dir: Directory to unpack the APK into.

    Returns:
        True if successful, False otherwise.
    """
    try:
        subprocess.run(["apktool", "d", "-f", apk_path, "-o", output_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"Error: APKtool failed to unpack {apk_path}")
        return False

def extract_static_features(decompiled_apk_dir: str) -> StaticFeatures:
    """
    Extracts static features from decompiled APK files.
    """
    # ... (old one's extract_static_features function)
    manifest_path = os.path.join(decompiled_apk_dir, "AndroidManifest.xml")
    try:
        tree = etree.parse(manifest_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing AndroidManifest.xml: {e}")
        return StaticFeatures(permissions=[], apis=[], urls=[], components={}, manifest_info={})
    
    # Extract permissions
    permissions = [p.get('{http://schemas.android.com/apk/res/android}name') for p in root.findall('.//uses-permission')]
    
    # Extract components
    components = {
        'activities': [a.get('{http://schemas.android.com/apk/res/android}name') for a in root.findall('.//activity')],
        'services': [s.get('{http://schemas.android.com/apk/res/android}name') for s in root.findall('.//service')],
        'receivers': [r.get('{http://schemas.android.com/apk/res/android}name') for r in root.findall('.//receiver')],
        'providers': [p.get('{http://schemas.com/apk/res/android}name') for p in root.findall('.//provider')]
    }

    # Simplified URL extraction for demonstration
    urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', open(manifest_path, 'r', encoding='utf-8').read())
    
    manifest_info = {
        'package_name': root.get('package'),
        'version_code': root.get('{http://schemas.android.com/apk/res/android}versionCode'),
        'version_name': root.get('{http://schemas.android.com/apk/res/android}versionName'),
    }

    return StaticFeatures(permissions=permissions, apis=[], urls=urls, components=components, manifest_info=manifest_info)


def extract_binary_features(apk_path: str) -> BinaryFeatures:
    """
    Extracts binary features from the raw APK file.
    """
    # ... (old one's extract_binary_features function)
    
    api_calls_sequences = []
    raw_strings = []
    call_graph_summary = {}

    try:
        a_obj, d_obj, x_obj = a(apk_path)
        for method in d_obj.get_methods():
            for instruction in method.get_instructions():
                output_str = instruction.get_output()
                api_match = re.search(r'Landroid/(.*);->(.*)\(.*\)', output_str)
                if api_match:
                    api_calls_sequences.append(api_match.group(0))
        
        raw_strings = a_obj.get_strings()
        
    except Exception as e:
        print(f"Error extracting binary features with androguard: {e}")
    
    return BinaryFeatures(api_calls_sequences=api_calls_sequences, raw_strings=raw_strings, call_graph_summary={})