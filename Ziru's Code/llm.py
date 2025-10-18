# llm.py
import hashlib
from typing import Dict, List, Optional
from .types import StaticFeatures, LLMDescriptions

class LLMDescriber:
    DEFAULT_PROMPT = (
        "You are a security analyst. Given an Android feature name, write one sentence "
        "explaining why it might be risky or noteworthy in malware analysis. Be concise."
    )
    def __init__(self, provider: Optional[str]=None, model: Optional[str]=None):
        self.provider = (provider or "").lower()
        self.model = model
        self._openai_client = None
        if self.provider == "openai":
            try:
                import openai
                self._openai_client = openai.OpenAI()
            except Exception:
                self._openai_client = None

    @staticmethod
    def _hash_key(kind: str, value: str) -> str:
        hv = hashlib.sha1(f"{kind}:{value}".encode("utf-8")).hexdigest()[:16]
        return f"{kind}:{hv}"

    def _describe_openai(self, features: List[str]) -> Dict[str,str]:
        out={}
        if self._openai_client is None: return out
        sys_prompt = self.DEFAULT_PROMPT
        for feat in features:
            try:
                prompt = f"{sys_prompt}\nFeature: {feat}"
                resp = self._openai_client.chat.completions.create(
                    model=self.model or "gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You write short, factual analyses."},
                        {"role":"user","content":prompt},
                    ],
                    temperature=0.2, max_tokens=64
                )
                text = (resp.choices[0].message.content or "").strip()
                out[self._hash_key("feat",feat)] = text
            except Exception:
                pass
        return out

    @staticmethod
    def _fallback_desc(feat: str) -> str:
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

    def describe(self, static: StaticFeatures) -> LLMDescriptions:
        pool = static.permissions + static.api_calls + static.urls + static.uses_features
        by_feature = {}
        if self.provider == "openai":
            by_feature.update(self._describe_openai(pool))
        for feat in pool:
            key = self._hash_key("feat", feat)
            if key not in by_feature:
                by_feature[key] = self._fallback_desc(feat)
        return LLMDescriptions(by_feature=by_feature)
