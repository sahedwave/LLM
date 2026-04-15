# LLM From Scratch

This project is a minimal educational character-level language model built from scratch in PyTorch.

The current stage includes:

- character-level tokenization
- a decoder-only Transformer
- next-character prediction training
- probabilistic text generation

## Project Structure

```text
llm_from_scratch/
├── data/
│   ├── data.txt
│   ├── nuclear_qa.txt
│   └── phase45_dataset.jsonl
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   └── utils.py
├── generate.py
├── README.md
├── requirements.txt
└── train.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
python generate.py
```

## Phase 4.5 Dataset

If you generate structured Phase 4.5 records, place them in:

```text
llm_from_scratch/data/phase45_dataset.jsonl
```

Supported JSONL rows:

```json
{"text": "..."}
{"text": "...", "concept": "neutron_transport", "type": "definition"}
```

The training pipeline will:

- read the JSONL file automatically
- infer `concept` and `type` if they are missing
- optionally prepend `Concept: <concept>` during training
- include the records in the normal training corpus without changing the Transformer architecture

Expected behavior:

- `python3 train.py` trains the model and saves `model.pt`
- `python3 generate.py` loads the checkpoint and generates text from the seed `"nuclear reactor"`
