#!/usr/bin/env python3
"""
Upload BMW E30 M3 service manual dataset to HuggingFace Hub.

This script uploads the prepared JSONL dataset to your HuggingFace account,
making it ready for use with AutoTrain or direct training.

Usage:
    python scripts/08_upload_to_hf.py --repo your-username/bmw-e30-service-manual

Requirements:
    pip install datasets huggingface_hub
    huggingface-cli login  # First time only
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description='Upload BMW E30 dataset to HuggingFace Hub')
    parser.add_argument(
        '--repo',
        type=str,
        required=True,
        help='HuggingFace repo name (e.g., username/bmw-e30-service-manual)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private (default: public)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing dataset'
    )

    args = parser.parse_args()

    # Paths - use AutoTrain format if available
    data_dir = Path('data')
    train_autotrain = data_dir / 'hf_train_autotrain.jsonl'
    val_synthetic = data_dir / 'hf_val_synthetic.jsonl'

    # Use AutoTrain flat text format (required for AutoTrain compatibility)
    if train_autotrain.exists() and val_synthetic.exists():
        train_file = train_autotrain
        val_file = val_synthetic
        print("📝 Using AutoTrain format (flat text with synthetic validation)")
    elif train_autotrain.exists():
        train_file = train_autotrain
        val_file = data_dir / 'hf_val.jsonl'
        print("⚠️  Using AutoTrain training but original validation")
    else:
        train_file = data_dir / 'hf_train.jsonl'
        val_file = data_dir / 'hf_val.jsonl'
        print("⚠️  Using original nested format (may not work with AutoTrain)")

    # Verify files exist
    if not train_file.exists() or not val_file.exists():
        print("❌ Error: Dataset files not found!")
        print(f"   Expected: {train_file}")
        print(f"   Expected: {val_file}")
        print("\nRun: make hf_prep  # to generate dataset")
        return

    print(f"📦 Preparing to upload dataset to: {args.repo}")
    print(f"   Train: {train_file} ({train_file.stat().st_size // 1024} KB)")
    print(f"   Val:   {val_file} ({val_file.stat().st_size // 1024} KB)")
    print(f"   Private: {args.private}")

    # Load dataset
    print("\n🔄 Loading dataset files...")
    dataset = load_dataset('json', data_files={
        'train': str(train_file),
        'validation': str(val_file)
    })

    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Validation: {len(dataset['validation'])} examples")

    # Show sample
    print(f"\n📝 Sample example:")
    sample = dataset['train'][0]
    if 'text' in sample:
        # AutoTrain flat text format
        lines = sample['text'].split('\n', 1)
        if len(lines) >= 2:
            print(f"   {lines[0][:80]}...")
            print(f"   {lines[1][:80]}...")
        else:
            print(f"   {sample['text'][:160]}...")
    elif 'messages' in sample:
        # Old nested format
        if 'meta' in sample:
            print(f"   Task: {sample['meta']['task']}")
        print(f"   User: {sample['messages'][0]['content'][:80]}...")
        print(f"   Assistant: {sample['messages'][1]['content'][:80]}...")

    # Create repo if it doesn't exist
    print(f"\n🚀 Uploading to HuggingFace Hub: {args.repo}")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type='dataset',
            private=args.private,
            exist_ok=True
        )
        print(f"   ✅ Repository created/verified")
    except Exception as e:
        print(f"   ⚠️  Repository may already exist: {e}")

    # Push to hub
    print(f"   📤 Uploading dataset files...")
    dataset.push_to_hub(
        args.repo,
        private=args.private,
        commit_message="Upload BMW E30 M3 service manual dataset"
    )

    print(f"\n✅ Upload complete!")
    print(f"🔗 View at: https://huggingface.co/datasets/{args.repo}")

    print(f"\n📋 Next steps:")
    print(f"   1. Go to https://huggingface.co/autotrain")
    print(f"   2. Create new project → LLM Fine-tuning")
    print(f"   3. Select dataset: {args.repo}")
    print(f"   4. Choose base model: meta-llama/Llama-3.1-8B-Instruct")
    print(f"   5. Column mapping: text_column = 'text'")
    print(f"   6. Training: Use 'train' split for training, 'validation' for validation")
    print(f"   7. Click 'Train' (cost: ~$5-10)")


if __name__ == '__main__':
    main()
