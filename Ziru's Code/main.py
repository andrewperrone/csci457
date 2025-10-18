import os
import sys
import shutil
from typing import Dict, List, Optional, Tuple
from data_structures import ClassificationResult, StaticFeatures, LLMDescriptions, ArtemisTensor
from feature_extractor import unpack_apk, extract_static_features, extract_binary_features
from llm_tensor_builder import LLMDescriber, TensorBuilder
from model_report_generator import DualBranchCNN, render_report_html

def analyze_apk(
    apk_path: str,
    work_dir: str,
    cnn_weights_path: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    report_output: Optional[str] = None,
) -> Dict[str, object]:
    """
    End-to-end analysis of an APK file.
    """
    # 1. Unpack the APK
    unpacked_dir = os.path.join(work_dir, "decompiled_apk")
    if not unpack_apk(apk_path, unpacked_dir):
        print("APK unpacking failed. Aborting.")
        return {}

    # 2. Extract Features
    static = extract_static_features(unpacked_dir)
    binary = extract_binary_features(apk_path)
    
    # 3. Generate LLM Descriptions
    llm_describer = LLMDescriber(provider=llm_provider, model=llm_model)
    llm = llm_describer.describe(static)
    
    # 4. Build Tensors
    tensor_builder = TensorBuilder()
    tensors = tensor_builder.build(static, llm)
    
    # 5. Classify with CNN
    cnn_model = DualBranchCNN()
    if cnn_weights_path and os.path.exists(cnn_weights_path):
        cnn_model.load_weights(cnn_weights_path)
    result = cnn_model.predict_one(tensors)

    # 6. Generate Report
    saved_report = None
    if report_output:
        saved_report = render_report_html(
            output_path=report_output,
            result=result,
            static=static,
            llm=llm,
        )

    # Cleanup
    shutil.rmtree(work_dir)

    return {
        "apk_path": apk_path,
        "static": static,
        "binary": binary,
        "llm_descriptions": llm,
        "classification": result,
        "report_path": saved_report,
    }

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Artemis APK Analyzer")
    p.add_argument("apk", help="Path to APK file")
    p.add_argument("--work", default="artemis_workdir", help="Working directory for decode & temp")
    p.add_argument("--weights", default=None, help="Path to CNN weights (.weights.h5 or .keras)")
    p.add_argument("--llm-provider", default=None, choices=[None, "openai", "hf"], help="LLM provider (optional)")
    p.add_argument("--llm-model", default=None, help="LLM model name for provider (optional)")
    p.add_argument("--report", default="artemis_report.html", help="Output HTML report path")
    args = p.parse_args()

    analyze_apk(
        apk_path=args.apk,
        work_dir=args.work,
        cnn_weights_path=args.weights,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        report_output=args.report,
    )
