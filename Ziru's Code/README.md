# Artemis (Modular) — Quickstart

## Layout
- `extractor.py` — APK unpack + static/binary feature extraction
- `llm.py` — LLM descriptions (OpenAI optional, heuristic fallback)
- `tensorizer.py` — Feature hashing + text embeddings
- `model.py` — Dual-branch 1D CNN (train + inference)
- `report.py` — HTML report with Jinja2 or fallback builder
- `analyze_apk.py` — One-shot analysis runner
- `train_cnn.py` — Trainer on CSV dataset
- `dataset_sample.csv` — Example CSV (edit APK paths)
- `demo_train.sh` — Example install + train script

## Install
```bash
pip install -r requirements.txt          # your existing
pip install androguard==3.4.0rc2 lxml==5.3.0 jinja2==3.1.4
```

## Train
Edit `dataset_sample.csv` to real APK paths, then:
```bash
python -m artemis_modular.train_cnn --dataset artemis_modular/dataset_sample.csv --out cnn_model.weights.h5
```

## Analyze
```bash
python -m artemis_modular.analyze_apk path/to/app.apk --weights cnn_model.weights.h5 --report report.html
```
