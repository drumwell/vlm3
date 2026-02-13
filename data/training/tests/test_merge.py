"""Tests for data/training/merge.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path so we can import merge
sys.path.insert(0, str(Path(__file__).parent.parent))
from merge import (
    apply_weighting,
    copy_source_images,
    create_blank_image,
    inject_blank_image,
    load_config,
    load_jsonl,
    load_source,
    merge,
    write_jsonl,
    MergeConfig,
    SourceConfig,
)


# ============================================================================
# Fixtures
# ============================================================================

def make_record(source_type="service_manual", content_type="procedure",
                image=None, question="Q?", answer="A."):
    """Create a minimal VLM record."""
    record = {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source_type": source_type,
            "content_type": content_type,
            "qa_id": f"test-{id(question)}",
        },
    }
    if image:
        record["image"] = image
    return record


@pytest.fixture
def source_dirs(tmp_path):
    """Create mock source prepared/ directories with train and val data."""
    # Manual source (with images)
    manual_dir = tmp_path / "src" / "manual" / "prepared"
    manual_dir.mkdir(parents=True)
    manual_images = manual_dir / "images"
    manual_images.mkdir()

    manual_train = [
        make_record(image="images/page01.jpg", question="What is torque?", answer="42 Nm"),
        make_record(image="images/page02.jpg", question="What is gap?", answer="0.8mm"),
        make_record(image="images/page03.jpg", question="What is timing?", answer="6 BTDC"),
        make_record(image="images/page04.jpg", question="What oil?", answer="5W-30"),
    ]
    manual_val = [
        make_record(image="images/page05.jpg", question="Valve clearance?", answer="0.25mm"),
    ]

    write_jsonl(manual_train, manual_dir / "manual_train.jsonl")
    write_jsonl(manual_val, manual_dir / "manual_val.jsonl")

    # Create dummy image files
    from PIL import Image
    for i in range(1, 6):
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        img.save(manual_images / f"page0{i}.jpg", "JPEG")

    # Forum source (text-only, no images)
    forum_dir = tmp_path / "src" / "forum" / "prepared"
    forum_dir.mkdir(parents=True)

    forum_train = [
        make_record(source_type="forum", content_type="troubleshooting",
                    question="Why won't it start?", answer="Check fuel pump relay"),
        make_record(source_type="forum", content_type="maintenance",
                    question="Best coolant?", answer="BMW blue coolant"),
    ]
    forum_val = [
        make_record(source_type="forum", content_type="troubleshooting",
                    question="Rough idle?", answer="Check ICV"),
    ]

    write_jsonl(forum_train, forum_dir / "forum_train.jsonl")
    write_jsonl(forum_val, forum_dir / "forum_val.jsonl")

    return tmp_path


@pytest.fixture
def config_file(source_dirs):
    """Create a merge config.yaml pointing at mock sources."""
    config = {
        "sources": {
            "manual": {
                "path": str(source_dirs / "src" / "manual" / "prepared"),
                "weight": 0.8,
            },
            "forum": {
                "path": str(source_dirs / "src" / "forum" / "prepared"),
                "weight": 0.2,
            },
        },
        "upsample_to_balance": True,
        "shuffle_seed": 42,
    }
    config_path = source_dirs / "training" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


# ============================================================================
# Tests: load_jsonl
# ============================================================================

class TestLoadJsonl:
    def test_loads_records(self, source_dirs):
        path = source_dirs / "src" / "manual" / "prepared" / "manual_train.jsonl"
        records = load_jsonl(path)
        assert len(records) == 4
        assert records[0]["conversations"][0]["content"] == "What is torque?"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_jsonl(path) == []


# ============================================================================
# Tests: load_source
# ============================================================================

class TestLoadSource:
    def test_loads_both_splits(self, source_dirs):
        source = SourceConfig(
            name="manual",
            path=source_dirs / "src" / "manual" / "prepared",
            weight=0.8,
        )
        data = load_source(source)
        assert len(data["train"]) == 4
        assert len(data["val"]) == 1

    def test_missing_source_returns_empty(self, tmp_path):
        source = SourceConfig(
            name="missing",
            path=tmp_path / "nonexistent",
            weight=1.0,
        )
        data = load_source(source)
        assert data["train"] == []
        assert data["val"] == []


# ============================================================================
# Tests: blank image
# ============================================================================

class TestCreateBlankImage:
    def test_creates_blank_jpg(self, tmp_path):
        path = create_blank_image(tmp_path)
        assert path.exists()
        assert path.name == "blank.jpg"

        from PIL import Image
        img = Image.open(path)
        assert img.size == (28, 28)


# ============================================================================
# Tests: copy images
# ============================================================================

class TestCopySourceImages:
    def test_copies_images(self, source_dirs, tmp_path):
        source = SourceConfig(
            name="manual",
            path=source_dirs / "src" / "manual" / "prepared",
            weight=0.8,
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = copy_source_images(source, output_dir)
        assert count == 5
        assert (output_dir / "images" / "page01.jpg").exists()

    def test_dry_run_does_not_copy(self, source_dirs, tmp_path):
        source = SourceConfig(
            name="manual",
            path=source_dirs / "src" / "manual" / "prepared",
            weight=0.8,
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = copy_source_images(source, output_dir, dry_run=True)
        assert count == 5
        assert not (output_dir / "images").exists()

    def test_no_images_dir(self, source_dirs, tmp_path):
        source = SourceConfig(
            name="forum",
            path=source_dirs / "src" / "forum" / "prepared",
            weight=0.2,
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = copy_source_images(source, output_dir)
        assert count == 0


# ============================================================================
# Tests: inject_blank_image
# ============================================================================

class TestInjectBlankImage:
    def test_injects_for_missing_image(self):
        records = [
            {"conversations": [], "metadata": {}},
            {"conversations": [], "metadata": {}, "image": "images/foo.jpg"},
            {"conversations": [], "metadata": {}},
        ]
        updated, count = inject_blank_image(records)
        assert count == 2
        assert updated[0]["image"] == "images/blank.jpg"
        assert updated[1]["image"] == "images/foo.jpg"
        assert updated[2]["image"] == "images/blank.jpg"

    def test_no_text_only(self):
        records = [
            {"conversations": [], "metadata": {}, "image": "images/a.jpg"},
        ]
        _, count = inject_blank_image(records)
        assert count == 0


# ============================================================================
# Tests: weighting
# ============================================================================

class TestApplyWeighting:
    def test_upsamples_smaller_source(self):
        config = MergeConfig(
            sources=[
                SourceConfig("manual", Path("."), 0.8),
                SourceConfig("forum", Path("."), 0.2),
            ],
            upsample_to_balance=True,
            shuffle_seed=42,
        )
        source_records = {
            "manual": {
                "train": [make_record(question=f"mq{i}") for i in range(100)],
                "val": [make_record(question=f"mv{i}") for i in range(10)],
            },
            "forum": {
                "train": [make_record(question=f"fq{i}") for i in range(10)],
                "val": [make_record(question=f"fv{i}") for i in range(2)],
            },
        }
        combined = apply_weighting(source_records, config)

        # Manual has 100 records at weight 0.8, forum has 10 at weight 0.2
        # Scale = max(100/0.8, 10/0.2) = max(125, 50) = 125
        # Manual target = 125 * 0.8 = 100 (no upsampling)
        # Forum target = 125 * 0.2 = 25 (upsampled from 10)
        assert len(combined["train"]) == 100 + 25

    def test_no_upsampling_when_disabled(self):
        config = MergeConfig(
            sources=[
                SourceConfig("a", Path("."), 0.5),
                SourceConfig("b", Path("."), 0.5),
            ],
            upsample_to_balance=False,
            shuffle_seed=42,
        )
        source_records = {
            "a": {"train": [make_record() for _ in range(10)], "val": []},
            "b": {"train": [make_record() for _ in range(5)], "val": []},
        }
        combined = apply_weighting(source_records, config)
        assert len(combined["train"]) == 15  # Simple concat

    def test_single_source_no_change(self):
        config = MergeConfig(
            sources=[SourceConfig("manual", Path("."), 1.0)],
            upsample_to_balance=True,
            shuffle_seed=42,
        )
        records = [make_record() for _ in range(50)]
        source_records = {"manual": {"train": records, "val": []}}
        combined = apply_weighting(source_records, config)
        assert len(combined["train"]) == 50


# ============================================================================
# Tests: full merge
# ============================================================================

class TestMerge:
    def test_full_merge(self, config_file, source_dirs):
        output_dir = source_dirs / "output"
        output_dir.mkdir()

        result = merge(config_file, output_dir)

        assert result["status"] == "ok"
        assert result["train_count"] > 0
        assert result["val_count"] > 0
        assert result["images_copied"] == 5

        # Check output files exist
        train_path = output_dir / "merged_train.jsonl"
        val_path = output_dir / "merged_val.jsonl"
        assert train_path.exists()
        assert val_path.exists()

        # Check blank image exists
        assert (output_dir / "images" / "blank.jpg").exists()

        # Check that text-only records got blank image
        train_records = load_jsonl(train_path)
        for r in train_records:
            assert "image" in r  # Every record has an image field

        # Forum records should use blank.jpg
        forum_records = [r for r in train_records
                         if r.get("metadata", {}).get("source_type") == "forum"]
        for r in forum_records:
            assert r["image"] == "images/blank.jpg"

    def test_dry_run(self, config_file, source_dirs):
        output_dir = source_dirs / "output"
        output_dir.mkdir()

        result = merge(config_file, output_dir, dry_run=True)

        assert result["status"] == "ok"
        assert result["train_count"] > 0
        # No files written
        assert not (output_dir / "merged_train.jsonl").exists()

    def test_missing_source_graceful(self, source_dirs):
        """Config lists a source that doesn't exist — should skip gracefully."""
        import yaml

        config = {
            "sources": {
                "manual": {
                    "path": str(source_dirs / "src" / "manual" / "prepared"),
                    "weight": 0.8,
                },
                "nonexistent": {
                    "path": str(source_dirs / "src" / "nonexistent" / "prepared"),
                    "weight": 0.2,
                },
            },
            "upsample_to_balance": True,
            "shuffle_seed": 42,
        }
        config_path = source_dirs / "config_partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = source_dirs / "output"
        output_dir.mkdir(exist_ok=True)

        result = merge(config_path, output_dir)
        assert result["status"] == "ok"
        assert result["train_count"] == 4  # Only manual records
