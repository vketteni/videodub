#!/usr/bin/env python3
"""Migration script to move data from old structure to new structure."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import click


def find_old_data_files(old_directory: Path) -> List[Dict]:
    """Find and catalog old data files."""
    old_files = []

    # Look for old pipeline output structure
    if (old_directory / "pipeline_output").exists():
        pipeline_dir = old_directory / "pipeline_output"

        for item in pipeline_dir.iterdir():
            if item.is_dir() and item.name.startswith("video_"):
                video_id = item.name[6:]  # Remove "video_" prefix

                file_info = {"video_id": video_id, "old_path": item, "files": {}}

                # Check for various file types
                for file_type, patterns in {
                    "metadata": ["metadata.json", "pipeline_result.json"],
                    "translation": ["translation.json"],
                    "audio": ["translated_audio/", "*.mp3", "*.wav"],
                }.items():
                    for pattern in patterns:
                        matches = list(item.glob(pattern))
                        if matches:
                            file_info["files"][file_type] = matches

                old_files.append(file_info)

    # Look for individual files in root directory
    for pattern in ["*.py", "*.json", "*.log"]:
        matches = list(old_directory.glob(pattern))
        if matches:
            print(f"Found {len(matches)} {pattern} files in root directory")

    return old_files


def migrate_video_data(old_video_info: Dict, new_base_dir: Path) -> bool:
    """Migrate data for a single video."""
    video_id = old_video_info["video_id"]
    old_path = old_video_info["old_path"]

    print(f"  Migrating video {video_id}...")

    try:
        # Create new video directory
        new_video_dir = new_base_dir / f"video_{video_id}"
        new_video_dir.mkdir(parents=True, exist_ok=True)

        # Migrate metadata files
        if "metadata" in old_video_info["files"]:
            for metadata_file in old_video_info["files"]["metadata"]:
                if metadata_file.name == "metadata.json":
                    dest = new_video_dir / "metadata.json"
                elif metadata_file.name == "pipeline_result.json":
                    dest = new_video_dir / "processing_result.json"
                else:
                    dest = new_video_dir / metadata_file.name

                shutil.copy2(metadata_file, dest)
                print(f"    ✅ Copied {metadata_file.name}")

        # Migrate translation files
        if "translation" in old_video_info["files"]:
            for translation_file in old_video_info["files"]["translation"]:
                dest = new_video_dir / translation_file.name
                shutil.copy2(translation_file, dest)
                print(f"    ✅ Copied {translation_file.name}")

        # Migrate audio files
        if "audio" in old_video_info["files"]:
            audio_dest_dir = new_video_dir / "translated_audio"
            audio_dest_dir.mkdir(exist_ok=True)

            for audio_item in old_video_info["files"]["audio"]:
                if audio_item.is_dir():
                    # Copy entire directory
                    dest_dir = new_video_dir / audio_item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(audio_item, dest_dir)
                    print(f"    ✅ Copied audio directory {audio_item.name}")
                else:
                    # Copy individual audio file
                    dest = audio_dest_dir / audio_item.name
                    shutil.copy2(audio_item, dest)
                    print(f"    ✅ Copied audio file {audio_item.name}")

        return True

    except Exception as e:
        print(f"    ❌ Failed to migrate {video_id}: {e}")
        return False


def update_file_formats(new_base_dir: Path) -> None:
    """Update file formats to match new structure."""
    print("\n📝 Updating file formats...")

    for video_dir in new_base_dir.glob("video_*"):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name[6:]

        # Update processing result format if needed
        old_result_file = video_dir / "pipeline_result.json"
        new_result_file = video_dir / "processing_result.json"

        if old_result_file.exists() and not new_result_file.exists():
            try:
                with open(old_result_file, "r") as f:
                    old_data = json.load(f)

                # Convert to new format
                new_data = {
                    "video_id": old_data.get("video_id", video_id),
                    "status": old_data.get("status", "completed"),
                    "started_at": old_data.get("processing_started"),
                    "completed_at": old_data.get("processing_completed"),
                    "target_language": old_data.get("target_language"),
                    "tts_engine": "openai",  # Assume OpenAI for old data
                    "files": old_data.get("files", {}),
                    "errors": old_data.get("errors", []),
                    "metadata": old_data.get("metadata"),
                }

                with open(new_result_file, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, indent=2, ensure_ascii=False)

                print(f"  ✅ Updated processing result for {video_id}")

            except Exception as e:
                print(f"  ❌ Failed to update processing result for {video_id}: {e}")


def create_backup(old_directory: Path) -> Optional[Path]:
    """Create backup of old directory."""
    backup_path = old_directory.parent / f"{old_directory.name}_backup"

    try:
        if backup_path.exists():
            shutil.rmtree(backup_path)

        shutil.copytree(old_directory, backup_path)
        print(f"✅ Backup created at: {backup_path}")
        return backup_path

    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return None


@click.command()
@click.option(
    "--old-dir",
    type=click.Path(exists=True, path_type=Path),
    default=".",
    help="Directory containing old project structure",
)
@click.option(
    "--new-dir",
    type=click.Path(path_type=Path),
    default="./migrated_output",
    help="Directory for migrated data",
)
@click.option("--backup/--no-backup", default=True, help="Create backup of old data")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be migrated without actually doing it",
)
def main(old_dir: Path, new_dir: Path, backup: bool, dry_run: bool):
    """Migrate data from old project structure to new structure."""
    print("🔄 YouTube Translation Pipeline - Data Migration")
    print("=" * 50)

    if dry_run:
        print("🧪 DRY RUN MODE - No files will be modified")

    # Find old data
    print(f"🔍 Scanning for old data in: {old_dir}")
    old_videos = find_old_data_files(old_dir)

    if not old_videos:
        print("ℹ️  No old video data found to migrate")
        return

    print(f"📊 Found {len(old_videos)} videos to migrate")

    if dry_run:
        print("\n📋 Migration Plan:")
        for video_info in old_videos:
            print(f"  📹 {video_info['video_id']}")
            for file_type, files in video_info["files"].items():
                print(f"    {file_type}: {len(files)} files")
        return

    # Create backup if requested
    if backup:
        print(f"\n💾 Creating backup...")
        backup_path = create_backup(old_dir)
        if not backup_path:
            print("❌ Backup failed - aborting migration")
            return

    # Create new directory structure
    new_dir.mkdir(parents=True, exist_ok=True)

    # Migrate each video
    print(f"\n🚚 Migrating data to: {new_dir}")
    successful_migrations = 0

    for video_info in old_videos:
        if migrate_video_data(video_info, new_dir):
            successful_migrations += 1

    # Update file formats
    update_file_formats(new_dir)

    # Summary
    print(f"\n📊 Migration Summary:")
    print(f"  Total videos: {len(old_videos)}")
    print(f"  Successful: {successful_migrations}")
    print(f"  Failed: {len(old_videos) - successful_migrations}")
    print(f"  New location: {new_dir}")

    if backup:
        print(f"  Backup: {backup_path}")

    print("\n✅ Migration completed!")
    print("\n💡 Next steps:")
    print("  1. Test the new structure with the refactored pipeline")
    print("  2. Update any scripts that reference the old paths")
    print("  3. Remove old files if everything works correctly")


if __name__ == "__main__":
    main()
