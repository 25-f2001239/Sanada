# Training & Evaluation Scripts

- `scripts/train.py {counter,tagger,separator,all} [--smoke|--full]`: Unified trainer for `SourceCountCNN`, `GroupFineTaggerCNN`, and `IterativeMaskSeparator`.
- `scripts/evaluate.py {counter,benchmarks,profile,all} [--smoke|--full]`: Unified evaluation & latency profiler (`evaluation/*.json`).
