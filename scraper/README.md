# vBulletin Forum Scraper

A polite web scraper for archiving vBulletin forums, designed for E30 M3 knowledge preservation.

## Features

- **Rate limiting**: Randomized 1.5-2.5 second delays between requests
- **Checkpoint/resume**: Can stop and restart without losing progress
- **Structured storage**: Both raw HTML and parsed JSON data
- **Image downloading**: Downloads embedded images with post references

## Installation

```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Test on Small Subforum)

### Step 1: Discover Forum Structure

```bash
python scraper/01_discover_forums.py
```

This crawls the forum index and outputs `forum_archive/data/forums.json` with all forum IDs.

### Step 2: Find Your Target Forum

Look at `forums.json` to find the "No-Start" FAQ subforum ID. Example structure:
```json
{
  "forums": [
    {
      "forum_id": "5",
      "title": "FAQs",
      "subforums": [
        {"forum_id": "42", "title": "No-Start"}
      ]
    }
  ]
}
```

### Step 3: Scrape Thread Listings

```bash
# Replace 42 with actual forum ID
python scraper/02_scrape_threads.py --forum-id 42

# For testing, limit pages
python scraper/02_scrape_threads.py --forum-id 42 --max-pages 2
```

### Step 4: Scrape Post Content

```bash
# Scrape all threads in the forum
python scraper/03_scrape_posts.py --forum-id 42

# For testing, limit threads
python scraper/03_scrape_posts.py --forum-id 42 --max-threads 5
```

### Step 5: Download Images

```bash
# Download images from scraped posts
python scraper/04_download_images.py --forum-id 42

# Or all images
python scraper/04_download_images.py --all
```

## Convenience Runner

For quick testing:

```bash
# Discover forums first
python scraper/run_test_scrape.py --stage discover

# Then scrape specific forum
python scraper/run_test_scrape.py --stage threads --forum-id 42 --max-pages 2
python scraper/run_test_scrape.py --stage posts --forum-id 42 --max-threads 5
python scraper/run_test_scrape.py --stage images --forum-id 42
```

## Directory Structure

```
forum_archive/
├── raw/
│   ├── forums/          # Raw HTML of forum pages
│   ├── threads/         # Raw HTML of thread pages
│   └── images/          # Downloaded images
├── data/
│   ├── forums.json      # Forum hierarchy
│   ├── threads_<id>.jsonl   # Thread listings per forum
│   ├── posts_<id>.jsonl     # Posts per forum
│   └── image_manifest.json  # Image URL -> local path mapping
├── checkpoints/         # Resume state
│   ├── threads_<id>.json
│   ├── posts.json
│   └── images.json
└── logs/                # Timestamped log files
```

## Data Schemas

### Thread Record (`threads_<id>.jsonl`)
```json
{
  "thread_id": "12345",
  "title": "Help! Car won't start",
  "url": "https://example.com/threads/12345-help-car-wont-start",
  "forum_id": "42",
  "author": "username",
  "reply_count": 15,
  "view_count": 1234,
  "is_sticky": false,
  "is_locked": false
}
```

### Post Record (`posts_<id>.jsonl`)
```json
{
  "post_id": "67890",
  "thread_id": "12345",
  "author": "helper_user",
  "content_html": "<p>Check the fuel pump relay...</p>",
  "content_text": "Check the fuel pump relay...",
  "created_at": "2023-12-15T10:30:00",
  "post_number": 3,
  "is_first_post": false,
  "images": ["https://...image1.jpg", "https://...image2.jpg"],
  "thread_title": "Help! Car won't start"
}
```

## Configuration

Edit `scraper/scraper_config.yaml` to adjust:

- Rate limiting delays
- HTTP timeout and headers
- Storage paths
- Image download settings

## Resuming After Interruption

The scraper automatically resumes from the last checkpoint. To force a fresh start:

```bash
# Remove checkpoints
rm forum_archive/checkpoints/*.json

# Or use --no-resume flag (where available)
python scraper/03_scrape_posts.py --forum-id 42 --resume=False
```

## Scaling to Full Site

After validating on the No-Start subforum:

1. Scrape all threads from all forums:
   ```bash
   python scraper/02_scrape_threads.py --all
   ```

2. Scrape all posts:
   ```bash
   python scraper/03_scrape_posts.py --all
   ```

3. Download all images:
   ```bash
   python scraper/04_download_images.py --all
   ```

For ~590k posts, expect:
- ~300k+ requests (threads + posts)
- ~150-250 hours at 1.5-2.5s/request
- Consider running in tmux/screen sessions
- Check disk space for images (could be many GB)

## Troubleshooting

### 403 Forbidden
The forum may block scrapers. Try:
- Increase delay in config
- Check if forum requires login for viewing

### Connection Timeout
Increase `timeout_seconds` in config.

### Parser Not Finding Elements
vBulletin 6 HTML structure may vary. Check raw HTML in `forum_archive/raw/` and adjust selectors in `parser.py`.
