import os
import hashlib
import json
import subprocess
from typing import Dict, List, Optional
from data_structures import StaticFeatures, LLMDescriptions, ArtemisTensor
from tensorflow import constant as tf_constant # For building tensors
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Guarded imports for heavy dependencies
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

class LLMDescriber:
    """
    Generates semantic descriptions for features using an LLM.
    """
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", cache_path: str = "llm_cache.json"):
        # ... (old one's LLMDescriber class)
        
        self.provider = provider
        self.model = model
        self.cache_path = cache_path
        self._cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self._cache = json.load(f)
        
    def _describe_openai(self, features: StaticFeatures) -> str:
        if not HAS_OPENAI:
            return "OpenAI library not found. Cannot generate description."
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        prompt = f"Analyze the following Android application permissions and components. Identify any suspicious or potentially malicious behaviors based on them. Permissions: {features.permissions}. Components: {features.components}. Provide a concise summary of potential risks."
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return self._fallback_desc(features)

    def _fallback_desc(self, features: StaticFeatures) -> str:
        return f"Fallback analysis: Permissions include {', '.join(features.permissions)}. Suspicious URLs found: {', '.join(features.urls)}. This is a basic description due to LLM failure."

    def describe(self, features: StaticFeatures) -> LLMDescriptions:
        """Generates a description for a given set of static features."""
        # Check cache first
        features_hash = hashlib.sha256(str(features).encode()).hexdigest()
        if features_hash in self._cache:
            return LLMDescriptions(by_feature=self._cache[features_hash])

        # Generate new description
        description = self._describe_openai(features)
        
        # Update cache
        self._cache[features_hash] = {"summary": description}
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

        return LLMDescriptions(by_feature={"summary": description})

class TensorBuilder:
    """
    Converts features and LLM descriptions into tensors.
    """
    def __init__(self):
        # ... (old one's TensorBuilder class)
        
        self.sbert_model = None
        if HAS_SBERT:
            try:
                self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Failed to load SentenceTransformer model: {e}")
                self.sbert_model = None

    def build(self, static: StaticFeatures, llm: LLMDescriptions) -> ArtemisTensor:
        """Builds tensors from features and descriptions."""
        
        # Simplified example for building static tensor (using hash)
        combined_features = " ".join(static.permissions + static.urls)
        feature_vector = [int(hashlib.sha256(f.encode()).hexdigest(), 16) % 1000000 for f in combined_features.split()]
        static_tensor = tf_constant(feature_vector, dtype='float32')
        static_tensor = tf_constant(pad_sequences([static_tensor], maxlen=100, padding='post').tolist())

        # Build LLM tensor
        llm_tensor = None
        if self.sbert_model and "summary" in llm.by_feature:
            embedding = self.sbert_model.encode(llm.by_feature["summary"])
            llm_tensor = tf_constant(embedding, dtype='float32')
        else:
            print("Warning: SentenceTransformer not available. Using fallback for LLM tensor.")
            llm_tensor = tf_constant([0.0] * 384, dtype='float32') # 384 is the default dimension
            
        return ArtemisTensor(static_tensor=static_tensor, llm_tensor=llm_tensor)