#!/usr/bin/env python3
"""
CLI tool to redistribute an ImageFolder dataset into train/val/test splits.
Assumes a directory structure with class subfolders, potentially already split into 'train' and 'val'.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of the dataset (should contain train/val or class folders)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Ratio of data for training"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Ratio of data for validation"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Ratio of data for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the split without moving files",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the split (move everything back to 'train')",
    )

    return parser.parse_args()


def get_all_images(data_dir: Path):
    """
    Find all images in the dataset, regardless of current structure.
    Returns a dict mapping class_name -> list of absolute file paths.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images_by_class = defaultdict(list)

    # Walk through the directory
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            # Determine class name: it's the name of the parent folder (or parent's parent if in train/val)
            # Strategy: look for the folder that is a direct child of 'train', 'val', 'test' OR data_dir

            parts = path.relative_to(data_dir).parts

            # If structure is data_dir/class/image.jpg -> parts=(class, image)
            # If structure is data_dir/train/class/image.jpg -> parts=(train, class, image)

            if len(parts) >= 2:
                # Check known split names
                if parts[0] in ["train", "val", "test", "validation", "testing"]:
                    if len(parts) >= 3:
                        class_name = parts[1]
                    else:
                        continue  # Skip top level files in split folders
                else:
                    class_name = parts[0]

                images_by_class[class_name].append(path)

    return images_by_class


def undo_split(data_dir: Path, dry_run: bool):
    """
    Move all files back to data_dir/train/class_name.
    """
    print(f"\nReverting split in {data_dir}...")
    images_by_class = get_all_images(data_dir)

    train_dir = data_dir / "train"

    for class_name, files in images_by_class.items():
        dest_class_dir = train_dir / class_name
        if not dry_run:
            dest_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Moving {len(files)} images to {dest_class_dir}...")

        for file_path in files:
            dest_path = dest_class_dir / file_path.name

            # Avoid overwriting if file already exists at dest (e.g. from previous bad runs)
            if dest_path.exists() and dest_path != file_path:
                print(f"    WARNING: {dest_path} exists. Skipping {file_path}")
                continue

            if not dry_run and file_path != dest_path:
                shutil.move(str(file_path), str(dest_path))

    # Clean up empty dirs
    if not dry_run:
        for split in ["val", "test"]:
            split_dir = data_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        print("  Removed empty 'val' and 'test' directories.")

    print("Undo complete.")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        sys.exit(1)

    if args.undo:
        undo_split(data_dir, args.dry_run)
        return

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)

    random.seed(args.seed)

    print(f"Scanning images in {data_dir}...")
    images_by_class = get_all_images(data_dir)
    class_names = sorted(images_by_class.keys())

    if not class_names:
        print("No images found! Check directory structure.")
        sys.exit(1)

    print(f"Found {len(class_names)} classes: {class_names}")
    total_images = sum(len(imgs) for imgs in images_by_class.values())
    print(f"Total images: {total_images}")

    # Prepare destination directories
    splits = ["train", "val", "test"]

    if args.dry_run:
        print("\n--- DRY RUN (No files will be moved) ---")

    print("\nProposed Split:")

    for class_name in class_names:
        files = images_by_class[class_name]
        # Sort to ensure deterministic shuffle with seed
        files.sort()
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * args.train_ratio)
        n_val = int(n_total * args.val_ratio)
        n_test = n_total - n_train - n_val

        split_files = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        print(
            f"  {class_name:<20}: {n_total} -> {len(split_files['train'])} train, {len(split_files['val'])} val, {len(split_files['test'])} test"
        )

        if not args.dry_run:
            for split, file_list in split_files.items():
                dest_dir = data_dir / split / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for file_path in file_list:
                    dest_path = dest_dir / file_path.name
                    if file_path != dest_path:
                        shutil.move(str(file_path), str(dest_path))

    if not args.dry_run:
        print("\nSuccess! Dataset split complete.")
        # Check validation
        print("Verifying counts...")
        for split in splits:
            split_dir = data_dir / split
            if split_dir.exists():
                count = len(list(split_dir.rglob("*")))
                # Only count files, not dirs
                file_count = sum(1 for p in split_dir.rglob("*") if p.is_file())
                print(f"  {split}: {file_count} files")
    else:
        print("\nDry run complete. No files moved.")


if __name__ == "__main__":
    main()
