#!/usr/bin/env python3
"""
Modal training app for Qwen2-VL-7B fine-tuning with LoRA.

Loads dataset directly from HuggingFace repo (JSONL + images format from 09_upload_vlm.py).

Prerequisites:
    1. Modal account: https://modal.com
    2. Modal CLI: pip install modal && modal setup
    3. HuggingFace secret: modal secret create huggingface HF_TOKEN=your_token
    4. Dataset uploaded: make upload (runs 09_upload_vlm.py)

Usage:
    # Dev training run (100 samples)
    modal run training/modal_train.py --max-samples 100

    # Full training
    modal run training/modal_train.py

    # Custom dataset repo
    modal run training/modal_train.py --dataset-repo username/vlm3
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import modal

# Modal configuration
GPU_TYPE = "A100-80GB"  # or "A100" for 40GB, "H100" for faster
TIMEOUT_HOURS = 24  # Modal max is 86400s (24h)
VOLUME_NAME = "vlm3-checkpoints"

# Training defaults (can be overridden via CLI)
DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen2-VL-7B-Instruct",
    "dataset_repo": "drumwell/vlm3",  # Your uploaded dataset
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "max_length": 2048,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 10,
}

# Docker image with all training dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",  # Required for Qwen2-VL processor
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.18.0",
        "qwen-vl-utils>=0.0.8",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.21.0",
    )
    .run_commands(
        # Install flash-attention for faster training (optional, may fail on some setups)
        "pip install flash-attn --no-build-isolation || echo 'flash-attn not installed, continuing without it'"
    )
)

# Create Modal app
app = modal.App("vlm3-training")

# Persistent volume for checkpoints
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Log file path (on the persistent volume)
LOG_DIR = Path("/checkpoints/logs")
LOG_FILE = None  # Set during training


def setup_logging():
    """Setup logging to both console and persistent file."""
    global LOG_FILE
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = LOG_DIR / f"training_{timestamp}.log"

    # Also create a symlink to latest log
    latest_link = LOG_DIR / "latest.log"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(LOG_FILE.name)

    return LOG_FILE


def log(message: str, also_print: bool = True):
    """Log message to file and optionally to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"

    if also_print:
        print(log_line)

    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(log_line + "\n")
            f.flush()  # Ensure immediate write


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            log(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
    except Exception as e:
        log(f"{prefix}GPU Memory: Unable to read ({e})")


def load_dataset_from_repo(repo_id: str, token: str, max_samples: int | None = None):
    """
    Load dataset from HuggingFace repo in JSONL + images format.

    Expected repo structure (from 09_upload_vlm.py):
        train.jsonl
        val.jsonl
        images/
            image1.jpg
            image2.jpg
            ...

    JSONL format:
        {"image": "images/xxx.jpg", "conversations": [...], "metadata": {...}}
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    from PIL import Image
    import tempfile

    print(f"Loading dataset from {repo_id}...")

    # Download JSONL files
    train_path = hf_hub_download(
        repo_id=repo_id,
        filename="train.jsonl",
        repo_type="dataset",
        token=token,
    )
    val_path = hf_hub_download(
        repo_id=repo_id,
        filename="val.jsonl",
        repo_type="dataset",
        token=token,
    )

    # Get list of image files
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    image_files = [f for f in all_files if f.startswith("images/")]
    print(f"Found {len(image_files)} images in repo")

    # Create a cache directory for images
    cache_dir = Path(tempfile.mkdtemp(prefix="vlm3_images_"))

    def load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def download_image(image_path: str) -> Image.Image | None:
        """Download and load image from HuggingFace."""
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=image_path,
                repo_type="dataset",
                token=token,
                cache_dir=str(cache_dir),
            )
            return Image.open(local_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return None

    # Load records
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    if max_samples:
        train_records = train_records[:max_samples]
        val_records = val_records[:max(1, max_samples // 10)]

    print(f"Loaded {len(train_records)} train, {len(val_records)} val records")

    return {
        "train": train_records,
        "val": val_records,
        "download_image": download_image,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_HOURS * 3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/checkpoints": volume},
)
def train(
    dataset_repo: str = DEFAULT_CONFIG["dataset_repo"],
    output_repo: str | None = None,
    max_samples: int | None = None,
    resume: bool = False,
    config_overrides: dict | None = None,
):
    """
    Fine-tune Qwen2-VL-7B with LoRA.

    Args:
        dataset_repo: HuggingFace dataset repo (e.g., "drumwell/vlm3")
        output_repo: HuggingFace repo to push adapter (e.g., "username/vlm3-lora")
        max_samples: Limit training samples (for dev runs)
        resume: Resume from latest checkpoint
        config_overrides: Override default training config
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    from torch.utils.data import Dataset as TorchDataset
    from huggingface_hub import HfApi

    # Setup persistent logging
    log_file = setup_logging()
    log(f"=== Training session started ===")
    log(f"Log file: {log_file}")

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")

    # Merge config
    config = {**DEFAULT_CONFIG, **(config_overrides or {})}

    log("=" * 60)
    log("VLM3 Training - Qwen2-VL-7B LoRA Fine-tuning")
    log("=" * 60)
    log(f"Dataset: {dataset_repo}")
    log(f"Base model: {config['base_model']}")
    log(f"GPU: {GPU_TYPE}")
    log(f"Max samples: {max_samples or 'all'}")
    log(f"LoRA r={config['lora_r']}, alpha={config['lora_alpha']}")
    log("=" * 60)

    # Load dataset
    log("\n📦 Loading dataset...")
    try:
        dataset_data = load_dataset_from_repo(dataset_repo, hf_token, max_samples)
        train_records = dataset_data["train"]
        val_records = dataset_data["val"]
        download_image = dataset_data["download_image"]
        log(f"Train samples: {len(train_records)}")
        log(f"Val samples: {len(val_records)}")
    except Exception as e:
        log(f"❌ DATASET LOADING FAILED: {type(e).__name__}: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        volume.commit()
        raise

    # Configure quantization
    log("\n🔧 Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and processor
    log("\n🤖 Loading Qwen2-VL-7B...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        log("Model loaded successfully")
        log_gpu_memory(prefix="Post-model-load ")

        processor = AutoProcessor.from_pretrained(
            config["base_model"],
            trust_remote_code=True,
        )
        log("Processor loaded successfully")
    except Exception as e:
        log(f"❌ MODEL LOADING FAILED: {type(e).__name__}: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        volume.commit()
        raise

    # Prepare model for training
    log("\n🔧 Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    log_gpu_memory(prefix="Post-kbit-prep ")

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create PyTorch Dataset
    class VLMDataset(TorchDataset):
        def __init__(self, records, processor, download_image_fn):
            self.processor = processor
            self.download_image = download_image_fn

            # Pre-download all images and filter out failures
            # VLM training requires images - text-only samples shouldn't be in the dataset
            print("Pre-loading images...")
            self.images = {}
            valid_records = []
            failed_count = 0

            for i, record in enumerate(records):
                image_path = record.get("image", "")
                if image_path:
                    img = download_image_fn(image_path)
                    if img is not None:
                        # Use new index for valid records
                        new_idx = len(valid_records)
                        self.images[new_idx] = img
                        valid_records.append(record)
                    else:
                        failed_count += 1
                else:
                    failed_count += 1

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(records)} images")

            self.records = valid_records
            print(f"  Done: {len(self.images)} valid, {failed_count} skipped")

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            record = self.records[idx]
            image = self.images[idx]  # Always exists now (filtered in __init__)

            # Build conversation for Qwen2-VL
            conversations = record.get("conversations", [])
            messages = []

            for i, conv in enumerate(conversations):
                role = conv["role"]
                content = conv["content"]

                if role == "user" and i == 0:
                    # First user message with image
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": content}
                        ]
                    })
                else:
                    messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })

            # Process with Qwen2-VL processor - no truncation to preserve image tokens
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,  # Will pad in collate_fn
            )

            # Remove batch dimension for text tensors, keep image tensors as-is
            item = {}
            for k, v in inputs.items():
                if k in ["input_ids", "attention_mask"]:
                    item[k] = v.squeeze(0)  # Remove batch dim for text
                elif k in ["pixel_values", "image_grid_thw"]:
                    item[k] = v  # Keep batch dim for images (will concat in collate)
                else:
                    item[k] = v.squeeze(0) if v.dim() > 1 else v

            item["labels"] = item["input_ids"].clone()

            return item

    # Create datasets (VLMDataset filters out records with failed images)
    log("\n📊 Creating training dataset...")
    try:
        train_dataset = VLMDataset(train_records, processor, download_image)
        log(f"Training dataset created: {len(train_dataset)} samples")
        log_gpu_memory(prefix="Post-train-dataset ")

        if val_records:
            log("Creating validation dataset...")
            val_dataset = VLMDataset(val_records, processor, download_image)
            log(f"Validation dataset created: {len(val_dataset)} samples")
        else:
            val_dataset = None
    except Exception as e:
        log(f"❌ DATASET CREATION FAILED: {type(e).__name__}: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        volume.commit()
        raise

    # Report actual counts after filtering
    train_skipped = len(train_records) - len(train_dataset)
    log(f"Training with {len(train_dataset)} samples ({train_skipped} skipped due to missing images)")
    if val_dataset:
        val_skipped = len(val_records) - len(val_dataset)
        log(f"Validation with {len(val_dataset)} samples ({val_skipped} skipped)")

    # Data collator with dynamic padding for Qwen2-VL
    def collate_fn(examples):
        # Get the pad token id
        pad_token_id = processor.tokenizer.pad_token_id or 0

        # Find max length in batch for text sequences
        max_len = max(ex["input_ids"].shape[0] for ex in examples)

        batch = {}
        for key in examples[0].keys():
            if key in ["input_ids", "attention_mask", "labels"]:
                # Pad sequences to max length in batch
                padded = []
                for ex in examples:
                    seq = ex[key]
                    pad_len = max_len - seq.shape[0]
                    if pad_len > 0:
                        if key == "labels":
                            # Pad labels with -100 (ignore in loss)
                            pad_val = -100
                        elif key == "attention_mask":
                            pad_val = 0
                        else:
                            pad_val = pad_token_id
                        seq = torch.cat([seq, torch.full((pad_len,), pad_val, dtype=seq.dtype)])
                    padded.append(seq)
                batch[key] = torch.stack(padded)
            elif key == "pixel_values":
                # Qwen2-VL uses variable-size pixel_values per image
                # Concatenate along first dimension, image_grid_thw tracks boundaries
                batch[key] = torch.cat([ex[key] for ex in examples], dim=0)
            elif key == "image_grid_thw":
                # Stack image grid info (should be same structure)
                batch[key] = torch.cat([ex[key] for ex in examples], dim=0)
            else:
                # For other keys, try to stack or concatenate
                try:
                    # Check if shapes match
                    shapes = [ex[key].shape for ex in examples]
                    if all(s == shapes[0] for s in shapes):
                        batch[key] = torch.stack([ex[key] for ex in examples])
                    else:
                        batch[key] = torch.cat([ex[key] for ex in examples], dim=0)
                except:
                    pass  # Skip keys that can't be batched
        return batch

    # Training arguments
    output_dir = "/checkpoints/vlm3-lora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config["max_grad_norm"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config["eval_steps"] if val_dataset else None,
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        fp16=False,
        bf16=True,
        dataloader_num_workers=0,  # Use 0 since we pre-load images
        remove_unused_columns=False,
        report_to="none",
    )

    # Custom callback for detailed logging
    class LoggingCallback(TrainerCallback):
        def __init__(self):
            self.last_log_step = 0

        def on_step_end(self, args, state, control, **kwargs):
            # Log every 10 steps
            if state.global_step % 10 == 0 and state.global_step != self.last_log_step:
                self.last_log_step = state.global_step
                log(f"Step {state.global_step}/{state.max_steps} (epoch {state.epoch:.2f})")
                log_gpu_memory(prefix="  ")

        def on_evaluate(self, args, state, control, **kwargs):
            log(f"Starting evaluation at step {state.global_step}...")
            log_gpu_memory(prefix="  Pre-eval ")
            # Commit logs before eval (in case it crashes)
            volume.commit()

        def on_save(self, args, state, control, **kwargs):
            log(f"Checkpoint saved at step {state.global_step}")
            log_gpu_memory(prefix="  ")
            # Commit logs after save
            volume.commit()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Log training metrics
                metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
                log(f"Metrics: {metrics_str}")

    # Initialize trainer
    log("\n🏋️ Starting training...")
    print("\n🏋️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[LoggingCallback()],
    )

    # Train with exception handling
    try:
        log_gpu_memory(prefix="Pre-training ")
        log(f"Training config: epochs={config['epochs']}, batch_size={config['batch_size']}, "
            f"grad_accum={config['gradient_accumulation_steps']}, lr={config['learning_rate']}")

        if resume:
            log("Resuming from checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        log("Training loop completed successfully")

    except Exception as e:
        log(f"❌ TRAINING FAILED: {type(e).__name__}: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        log_gpu_memory(prefix="At crash ")

        # Commit logs so we can see what happened
        volume.commit()

        # Re-raise to signal failure
        raise

    # Save final adapter
    log("\n💾 Saving adapter...")
    print("\n💾 Saving adapter...")
    final_adapter_path = f"{output_dir}/final"
    trainer.save_model(final_adapter_path)
    processor.save_pretrained(final_adapter_path)

    # Commit volume changes
    volume.commit()

    # Push to Hub if requested
    if output_repo:
        log(f"\n📤 Pushing adapter to HuggingFace Hub: {output_repo}")
        print(f"\n📤 Pushing adapter to HuggingFace Hub: {output_repo}")
        api = HfApi()
        api.upload_folder(
            folder_path=final_adapter_path,
            repo_id=output_repo,
            repo_type="model",
            token=hf_token,
            commit_message="Upload Qwen2-VL LoRA adapter from VLM3 training",
        )
        log(f"✅ Adapter pushed to: https://huggingface.co/{output_repo}")
        print(f"✅ Adapter pushed to: https://huggingface.co/{output_repo}")

    log("\n✅ Training complete!")
    print("\n✅ Training complete!")
    log(f"Adapter saved to: {final_adapter_path}")
    print(f"Adapter saved to: {final_adapter_path}")

    return {
        "status": "success",
        "adapter_path": final_adapter_path,
        "train_samples": len(train_records),
        "val_samples": len(val_records) if val_records else 0,
        "output_repo": output_repo,
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/checkpoints": volume},
)
def check_logs(tail: int = 100):
    """Read training logs from the persistent volume."""
    log_dir = Path("/checkpoints/logs")

    if not log_dir.exists():
        return {"status": "no_logs", "message": "No logs directory found"}

    # List all log files
    log_files = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        return {"status": "no_logs", "message": "No log files found"}

    # Get the latest log
    latest = log_files[0]
    with open(latest, "r") as f:
        lines = f.readlines()

    return {
        "status": "ok",
        "log_file": latest.name,
        "total_lines": len(lines),
        "last_lines": "".join(lines[-tail:]) if tail else "".join(lines),
        "all_logs": [f.name for f in log_files],
    }


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_CONFIG["dataset_repo"],
    output_repo: str = None,
    max_samples: int = None,
    resume: bool = False,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
):
    """
    VLM3 Training - Fine-tune Qwen2-VL-7B with LoRA on Modal.

    Examples:
        # Dev run (100 samples)
        modal run training/modal_train.py --max-samples 100

        # Full training with default dataset (drumwell/vlm3)
        modal run training/modal_train.py

        # Train and push adapter to Hub
        modal run training/modal_train.py --output-repo username/vlm3-lora

        # Custom dataset repo
        modal run training/modal_train.py --dataset-repo username/my-vlm-dataset

        # Check training logs (useful after a crash)
        modal run training/modal_train.py::check_logs_cli
    """
    # Build config overrides
    config_overrides = {}
    if epochs:
        config_overrides["epochs"] = epochs
    if batch_size:
        config_overrides["batch_size"] = batch_size
    if learning_rate:
        config_overrides["learning_rate"] = learning_rate

    print(f"Starting training job on Modal...")
    print(f"Dataset: {dataset_repo}")
    print(f"Max samples: {max_samples or 'all'}")

    # Run training
    result = train.remote(
        dataset_repo=dataset_repo,
        output_repo=output_repo,
        max_samples=max_samples,
        resume=resume,
        config_overrides=config_overrides if config_overrides else None,
    )

    print("\n" + "=" * 60)
    print("Training Result")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def check_logs_cli(tail: int = 100):
    """
    Check training logs from the persistent volume.

    Usage:
        modal run training/modal_train.py::check_logs_cli
        modal run training/modal_train.py::check_logs_cli --tail 50
    """
    print("Fetching logs from Modal volume...")
    result = check_logs.remote(tail=tail)

    if result["status"] == "no_logs":
        print(f"⚠️  {result['message']}")
        return

    print(f"\n📋 Log file: {result['log_file']}")
    print(f"📊 Total lines: {result['total_lines']}")
    print(f"📁 All log files: {', '.join(result['all_logs'])}")
    print("\n" + "=" * 60)
    print(f"Last {tail} lines:")
    print("=" * 60)
    print(result["last_lines"])
