# Google Drive Upload Automation Setup

## Quick Start

The Makefile now includes `make gdrive_upload` to automatically upload your dataset files to Google Drive.

## One-Time Setup (rclone)

### 1. Install rclone

**macOS**:
```bash
brew install rclone
```

**Linux**:
```bash
curl https://rclone.org/install.sh | sudo bash
```

**Windows**:
Download from https://rclone.org/downloads/

### 2. Configure Google Drive

Run the configuration wizard:
```bash
rclone config
```

Follow these steps:
1. **n** (New remote)
2. **Name**: `gdrive` (exactly this - used in Makefile)
3. **Storage**: Enter number for "Google Drive" (usually `15` or `drive`)
4. **client_id**: Press Enter (leave blank)
5. **client_secret**: Press Enter (leave blank)
6. **scope**: Enter `1` (Full access)
7. **root_folder_id**: Press Enter (leave blank)
8. **service_account_file**: Press Enter (leave blank)
9. **Edit advanced config**: `n` (No)
10. **Auto config**: `y` (Yes)
11. Browser will open - authorize rclone access to Google Drive
12. **Configure as team drive**: `n` (No)
13. **Keep this remote**: `y` (Yes)
14. **Quit config**: `q` (Quit)

### 3. Verify Setup

Test that rclone can access your Drive:
```bash
rclone lsd gdrive:
```

You should see a list of your Google Drive folders.

### 4. Create Target Directories (First Time Only)

```bash
rclone mkdir gdrive:llm3
rclone mkdir gdrive:llm3/data
rclone mkdir gdrive:llm3/notebooks
```

This creates the structure:
```
/MyDrive/llm3/
├── data/
└── notebooks/
```

## Usage

Once configured, uploading is automatic:

```bash
# Upload all dataset files to Google Drive
make gdrive_upload
```

This uploads:
- `data/hf_train.jsonl` → `/MyDrive/llm3/data/`
- `data/hf_val.jsonl` → `/MyDrive/llm3/data/`
- `config.yaml` → `/MyDrive/llm3/`
- `notebooks/finetune_qlora.ipynb` → `/MyDrive/llm3/notebooks/`
- `notebooks/test_inference.ipynb` → `/MyDrive/llm3/notebooks/`

## What Gets Uploaded

| Local File | Google Drive Location |
|------------|----------------------|
| `data/hf_train.jsonl` (778KB) | `/MyDrive/llm3/data/hf_train.jsonl` |
| `data/hf_val.jsonl` (91KB) | `/MyDrive/llm3/data/hf_val.jsonl` |
| `config.yaml` (3KB) | `/MyDrive/llm3/config.yaml` |
| `notebooks/finetune_qlora.ipynb` (17KB) | `/MyDrive/llm3/notebooks/finetune_qlora.ipynb` |
| `notebooks/test_inference.ipynb` (21KB) | `/MyDrive/llm3/notebooks/test_inference.ipynb` |

**Total upload size**: ~910KB (very quick!)

## Typical Workflow

```bash
# 1. Regenerate dataset (if needed)
make hf_prep

# 2. Upload everything to Google Drive
make gdrive_upload

# 3. Open notebook from Google Drive in Colab
# Option A: Go to https://colab.research.google.com/
#           File → Open notebook → Google Drive tab
#           Navigate to /MyDrive/llm3/notebooks/finetune_qlora.ipynb
#
# Option B: In Google Drive, right-click notebook → Open with → Google Colaboratory
#
# Files are already mounted at /MyDrive/llm3/ - just run the cells!
```

## Manual Upload Alternative

If you prefer not to use rclone, you can still manually upload via:
1. Go to https://drive.google.com
2. Create folder structure: `/MyDrive/llm3/data/`
3. Drag and drop the 3 files

But `make gdrive_upload` is faster for iterations! 😄

## Troubleshooting

### "command not found: rclone"
Install rclone first (see step 1)

### "Failed to create file system for gdrive"
Run `rclone config` to set up the remote

### "directory not found"
Create the target directory:
```bash
rclone mkdir gdrive:llm3
rclone mkdir gdrive:llm3/data
```

### Files not appearing in Drive
Check you authorized the correct Google account:
```bash
rclone about gdrive:
```

## Advanced Usage

### Upload with sync (delete old files)
```bash
rclone sync data/hf_train.jsonl gdrive:llm3/data/
```

### Upload entire data folder
```bash
rclone copy data/ gdrive:llm3/data/ --include "hf_*.jsonl"
```

### Check what would be uploaded (dry run)
```bash
rclone copy data/hf_train.jsonl gdrive:llm3/data/ --dry-run
```

## Benefits

- ✅ **Fast**: ~870KB uploads in seconds
- ✅ **Automated**: One command after dataset regeneration
- ✅ **Reliable**: Resume on interruption
- ✅ **Progress**: See upload progress in terminal
- ✅ **Incremental**: Only uploads changed files

## Fun Fact

The directory is named `llm3` as a pun on:
- **LLM** (Large Language Model)
- **E30 M3** (BMW model)
- **LLM³** (LLM cubed!) 😄
