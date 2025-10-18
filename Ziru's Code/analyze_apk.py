# analyze_apk.py
import os, json, time
from .types import ClassificationResult
from .extractor import unpack_apk, extract_static_features, extract_binary_features
from .llm import LLMDescriber
from .tensorizer import TensorBuilder
from .model import DualBranchCNN
from .report import render_report_html

def analyze_apk(apk_path: str, work_dir: str, weights: str=None,
                llm_provider: str=None, llm_model: str=None, report_path: str=None):
    os.makedirs(work_dir, exist_ok=True)
    decoded = os.path.join(work_dir, "decoded")
    if not unpack_apk(apk_path, decoded):
        raise RuntimeError("apktool unpack failed")

    static = extract_static_features(decoded)
    binary = extract_binary_features(apk_path)

    llm = LLMDescriber(provider=llm_provider, model=llm_model).describe(static)

    tb = TensorBuilder()
    tensor = tb.build(static, binary, llm)

    embed_dim = tensor.branch_b.shape[0]
    model = DualBranchCNN(hash_dim=tensor.branch_a.shape[0], embed_dim=embed_dim)
    if weights and os.path.exists(weights):
        model.load_weights(weights)
    else:
        print("[WARN] Using randomly initialized CNN (no weights supplied).")

    res = model.predict_one(tensor)

    saved = None
    if report_path:
        saved = render_report_html(report_path, res, static, binary, llm,
                                   generated_at=time.strftime("%Y-%m-%d %H:%M:%S"))

    return {
        "apk_path": apk_path,
        "classification": {"is_malicious": res.is_malicious, "confidence": res.confidence},
        "report_path": saved
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Artemis Modular APK Analyzer")
    p.add_argument("apk")
    p.add_argument("--work", default="work_one")
    p.add_argument("--weights", default=None)
    p.add_argument("--llm-provider", default=None, choices=[None,"openai","hf"])
    p.add_argument("--llm-model", default=None)
    p.add_argument("--report", default="artemis_report.html")
    args = p.parse_args()
    out = analyze_apk(args.apk, args.work, args.weights, args.llm_provider, args.llm_model, args.report)
    print(json.dumps(out, indent=2))
