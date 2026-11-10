# nanoLLM Homework 1

Experiments for Homework 1, **Language Models from Scratch**.

## Report

- [Technical report (Markdown)](HW1_report.md)
- [Report for browser printing/PDF export](HW1_report.html)

The report covers metric calculations, untrained tokenizer baselines, RNN/GRU/LSTM/Transformer comparisons, context length, and attention-head analysis.

## Reproduce

Create a Python 3.10+ environment and install the packages listed in the course tutorial. Then run:

```bash
python -m pytest nanolm/tests -q
python -m nanolm.train -c nanolm/configs/00_smoke.yaml
python -m nanolm.sweep \
  --sweep nanolm/configs/sweeps/01_char_models.yaml \
  --set train.compile=false
python -m nanolm.analyze \
  outputs/nanolm/sweeps/char_models \
  --baseline gpt_textbook \
  --metric val_bpb
```

The experiments use Tiny Shakespeare and save metrics, logs, summaries, and figures under `outputs/nanolm/`. Datasets, checkpoints, and virtual environments are intentionally excluded from Git.
