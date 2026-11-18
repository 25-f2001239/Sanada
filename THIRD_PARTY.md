# Third-Party Models & Datasets

EchoCount downloads third-party pretrained models and datasets separately via `scripts/download_models.py`. **Licences and training-dataset terms differ per model** — review each upstream repository before commercial deployment:

| Component | Upstream Source | Code/Weights Licence | Training Dataset / Terms Note |
| :--- | :--- | :--- | :--- |
| AudioSet 527-Class AST | `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause | Fine-tuned on AudioSet (YouTube metadata) |
| Open-Vocabulary CLAP | `laion/clap-htsat-unfused` | Apache-2.0 / MIT | Trained on LAION-Audio-630K |
| 2-Speaker Speech Separator (SepFormer) | `speechbrain/sepformer-wsj02mix` | Apache-2.0 | **Trained on WSJ0-2mix** (WSJ0 is governed by LDC/research-only terms) |
| Speaker Embedding Verifier (ECAPA-TDNN) | `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 | Trained on VoxCeleb1+2 |
| FSD50K Isolated Clips (subset) | `Fhrozen/FSD50k` / Zenodo `4060432` | CC-BY-4.0 / CC0 | Per-clip Creative Commons licences in `data/isolated_sources/manifest.json` |
