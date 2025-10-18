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

@dataclass
class StaticFeatures:
    """Class for holding extracted static features."""
    permissions: List[str]
    apis: List[str]
    urls: List[str]
    components: Dict[str, List[str]]
    manifest_info: Dict[str, object]

@dataclass
class BinaryFeatures:
    """Class for holding extracted binary features."""
    api_calls_sequences: List[List[str]]
    raw_strings: List[str]
    call_graph_summary: Dict[str, List[str]]

@dataclass
class LLMDescriptions:
    """Class for holding LLM-generated descriptions."""
    by_feature: Dict[str, str]

@dataclass
class ArtemisTensor:
    """Class for holding the final tensors for CNN input."""
    static_tensor: any # Use `any` for now, will be tf.Tensor
    llm_tensor: any # Use `any` for now, will be tf.Tensor

@dataclass
class ClassificationResult:
    """Class for holding the classification outcome."""
    label: str
    confidence: float
    is_malicious: bool
    details: Dict[str, object]