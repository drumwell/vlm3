# VLM Fine-tuning

Fine-tune Qwen2-VL-7B on the BMW E30 M3 service manual Q&A dataset using Modal + LoRA.

## Prerequisites

1. **Dataset uploaded**: Run `make upload` first (uses `pipeline/scripts/09_upload_vlm.py`)

2. **Create Modal account**: https://modal.com (free tier includes GPU credits)

3. **Set up Modal CLI**:
   ```bash
   pip install modal
   modal setup
   ```

4. **Configure HuggingFace secret in Modal**:
   ```bash
   modal secret create huggingface HF_TOKEN=your_token_here
   ```

## Quick Start

```bash
# Dev run first (100 samples, ~10 min, ~$0.50)
make train-dev

# Full training (~2-4 hours, ~$8-16)
make train

# Push trained adapter to HuggingFace
make train HF_MODEL_REPO=your-username/vlm3-lora
```

## Commands

| Command | Description |
|---------|-------------|
| `make train` | Full training on Modal (A100-80GB) |
| `make train-dev` | Dev run with 100 samples |
| `make train-resume` | Resume from checkpoint |

### Direct Modal Commands

```bash
# Dev run
modal run training/modal_train.py --max-samples 100

# Full training
modal run training/modal_train.py

# Custom dataset
modal run training/modal_train.py --dataset-repo username/my-dataset

# Custom epochs
modal run training/modal_train.py --epochs 5

# Push adapter to Hub
modal run training/modal_train.py --output-repo username/vlm3-lora
```

## Configuration

Training config (in `modal_train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | Qwen/Qwen2-VL-7B-Instruct | Base VLM |
| `dataset_repo` | drumwell/vlm3 | HuggingFace dataset |
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha |
| `lora_dropout` | 0.05 | LoRA dropout |
| `epochs` | 3 | Training epochs |
| `batch_size` | 4 | Per-device batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `learning_rate` | 2e-4 | Learning rate |
| `max_length` | 2048 | Max sequence length |

LoRA targets all projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Dataset Format

Expects HuggingFace repo with this structure (from `make upload`):

```
drumwell/vlm3/
├── train.jsonl
├── val.jsonl
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── README.md
```

JSONL format:
```json
{
  "image": "images/data_src_xxx.jpg",
  "conversations": [
    {"role": "user", "content": "What torque spec..."},
    {"role": "assistant", "content": "The torque is 40 Nm..."}
  ],
  "metadata": {"page_id": "...", "source_type": "..."}
}
```

## Estimated Costs

| Run Type | Duration | Cost (A100-80GB @ ~$3/hr) |
|----------|----------|---------------------------|
| Dev run (100 samples) | ~10 min | ~$0.50 |
| Full training (11k samples, 3 epochs) | 2-4 hours | $8-16 |

## Outputs

After training:
- **Checkpoints**: Saved to Modal volume `vlm3-checkpoints`
- **Final adapter**: `/checkpoints/vlm3-lora/final/`
- **HuggingFace** (optional): Push with `--output-repo`

## Troubleshooting

### Modal not finding HuggingFace token
```bash
modal secret create huggingface HF_TOKEN=your_token
```

### Out of memory
Reduce batch size:
```bash
modal run training/modal_train.py --batch-size 2
```

### Dataset not found
Ensure you've uploaded with `make upload` first.

## Directory Structure

```
training/
├── modal_train.py        # Modal training app
├── requirements.txt      # Local dev dependencies
├── configs/
│   └── lora_qwen2vl.yaml # LoRA config reference
└── README.md
```

## Next Steps

After training:
1. **Download adapter**: From Modal volume or HuggingFace
2. **Evaluate**: See `eval/` directory (coming soon)
3. **Merge adapter**: Use PEFT to merge with base model
4. **Deploy**: Create inference endpoint
