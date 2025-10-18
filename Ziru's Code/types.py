# types.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

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
    instruction_sequences: List[str]
    method_calls: List[str]

@dataclass
class LLMDescriptions:
    by_feature: Dict[str, str]

@dataclass
class ArtemisTensor:
    branch_a: np.ndarray
    branch_b: np.ndarray

@dataclass
class ClassificationResult:
    is_malicious: bool
    confidence: float
    logits: Optional[float] = None
