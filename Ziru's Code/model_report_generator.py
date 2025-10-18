import os
import io
import sys
import json
import shutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras.utils import plot_model
from typing import Dict, List, Optional
from data_structures import ClassificationResult, StaticFeatures, LLMDescriptions, ArtemisTensor

class DualBranchCNN:
    """
    A dual-branch 1D CNN for combined feature and text classification.
    """
    def __init__(self, static_vocab_size: int = 100000, llm_embedding_dim: int = 384):
        # ... (原有代码中 DualBranchCNN 类的内容)
        
        self.static_vocab_size = static_vocab_size
        self.llm_embedding_dim = llm_embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        # Branch 1: Static and Binary Features
        static_input = Input(shape=(100,), dtype='float32', name='static_input')
        static_conv = Conv1D(filters=64, kernel_size=5, activation='relu')(static_input)
        static_pool = GlobalMaxPooling1D()(static_conv)
        
        # Branch 2: LLM Descriptions
        llm_input = Input(shape=(self.llm_embedding_dim,), dtype='float32', name='llm_input')
        llm_dense = Dense(32, activation='relu')(llm_input)

        # Concatenate and classify
        merged = Concatenate()([static_pool, llm_dense])
        final_dense = Dense(16, activation='relu')(merged)
        output = Dense(1, activation='sigmoid')(final_dense)
        
        model = Model(inputs=[static_input, llm_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_weights(self, weights_path: str):
        self.model.load_weights(weights_path)

    def predict_one(self, tensors: ArtemisTensor) -> ClassificationResult:
        """Performs a single prediction."""
        try:
            prediction = self.model.predict([tensors.static_tensor[None, :], tensors.llm_tensor[None, :]])[0][0]
            label = "Malicious" if prediction > 0.5 else "Benign"
            return ClassificationResult(
                label=label,
                confidence=float(prediction) if label == "Malicious" else float(1 - prediction),
                is_malicious=prediction > 0.5,
                details={}
            )
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return ClassificationResult(label="Unknown", confidence=0.0, is_malicious=False, details={"error": str(e)})


def render_report_html(output_path: str, result: ClassificationResult, static: StaticFeatures, llm: LLMDescriptions) -> str:
    # ... (old one's render_report_html function)

    template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Artemis Analysis Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
            .container {{ width: 80%; margin: auto; padding: 20px; }}
            h1, h2, h3 {{ border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
            .label-malicious {{ color: #dc3545; font-weight: bold; }}
            .label-benign {{ color: #28a745; font-weight: bold; }}
            pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Artemis Analysis Report</h1>
        <h2>Summary</h2>
        <p><strong>Classification:</strong> <span class="label-{result.label.lower()}">{result.label}</span></p>
        <p><strong>Confidence:</strong> {result.confidence:.2%}</p>

        <h2>LLM-based Behavioral Analysis</h2>
        <pre>{llm.by_feature.get('summary', 'No LLM summary available.')}</pre>

        <h2>Static Feature Details</h2>
        <h3>Permissions</h3>
        <ul>{''.join([f'<li>{p}</li>' for p in static.permissions])}</ul>
        
        <h3>URLs Found</h3>
        <ul>{''.join([f'<li>{url}</li>' for url in static.urls])}</ul>
        
        <h3>Components</h3>
        <pre>{json.dumps(static.components, indent=2)}</pre>
        
    </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    return output_path