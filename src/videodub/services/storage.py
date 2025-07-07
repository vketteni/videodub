"""Storage service implementation."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
import structlog

from ..core.exceptions import StorageError
from ..core.interfaces import StorageService
from ..core.models import ProcessingResult, TranslationSegment, VideoMetadata

logger = structlog.get_logger(__name__)


class FileStorageService(StorageService):
    """File-based storage service implementation."""

    def __init__(self, base_directory: Path):
        """
        Initialize file storage service.

        Args:
            base_directory: Base directory for all storage
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "File storage service initialized", base_directory=str(self.base_directory)
        )

    async def save_metadata(self, video_id: str, metadata: VideoMetadata) -> Path:
        """
        Save video metadata to storage.

        Args:
            video_id: Video identifier
            metadata: Video metadata to save

        Returns:
            Path where metadata was saved
        """
        try:
            video_dir = await self.get_video_directory(video_id)
            metadata_path = video_dir / "metadata.json"

            # Convert metadata to dict
            metadata_dict = {
                "video_id": metadata.video_id,
                "title": metadata.title,
                "duration": metadata.duration,
                "url": metadata.url,
                "channel": metadata.channel,
                "upload_date": metadata.upload_date,
                "view_count": metadata.view_count,
                "description": metadata.description,
                "saved_at": datetime.now().isoformat(),
            }

            async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(metadata_dict, indent=2, ensure_ascii=False))

            logger.debug("Metadata saved", video_id=video_id, path=str(metadata_path))
            return metadata_path

        except Exception as e:
            logger.error("Failed to save metadata", video_id=video_id, error=str(e))
            raise StorageError(f"Failed to save metadata: {str(e)}")

    async def save_translation_data(
        self, video_id: str, segments: List[TranslationSegment]
    ) -> Path:
        """
        Save translation data to storage.

        Args:
            video_id: Video identifier
            segments: Translated segments to save

        Returns:
            Path where translation data was saved
        """
        try:
            video_dir = await self.get_video_directory(video_id)
            translation_path = video_dir / "translation.json"

            # Convert segments to dict
            translation_data = {
                "video_id": video_id,
                "target_language": segments[0].language if segments else None,
                "segment_count": len(segments),
                "saved_at": datetime.now().isoformat(),
                "segments": [
                    {
                        "start_time": seg.original_segment.start_time,
                        "end_time": seg.original_segment.end_time,
                        "original_text": seg.original_segment.text,
                        "translated_text": seg.translated_text,
                        "language": seg.language,
                        "audio_path": str(seg.audio_path) if seg.audio_path else None,
                    }
                    for seg in segments
                ],
            }

            async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(translation_data, indent=2, ensure_ascii=False)
                )

            logger.debug(
                "Translation data saved",
                video_id=video_id,
                segment_count=len(segments),
                path=str(translation_path),
            )
            return translation_path

        except Exception as e:
            logger.error(
                "Failed to save translation data", video_id=video_id, error=str(e)
            )
            raise StorageError(f"Failed to save translation data: {str(e)}")

    async def save_processing_result(self, result: ProcessingResult) -> Path:
        """
        Save processing result to storage.

        Args:
            result: Processing result to save

        Returns:
            Path where result was saved
        """
        try:
            video_dir = await self.get_video_directory(result.video_id)
            result_path = video_dir / "processing_result.json"

            # Convert result to dict
            result_dict = {
                "video_id": result.video_id,
                "status": result.status.value,
                "started_at": result.started_at.isoformat(),
                "completed_at": result.completed_at.isoformat()
                if result.completed_at
                else None,
                "target_language": result.target_language,
                "tts_engine": result.tts_engine.value if result.tts_engine else None,
                "files": result.files,
                "errors": result.errors,
                "metadata": {
                    "video_id": result.metadata.video_id,
                    "title": result.metadata.title,
                    "duration": result.metadata.duration,
                    "url": result.metadata.url,
                    "channel": result.metadata.channel,
                    "upload_date": result.metadata.upload_date,
                    "view_count": result.metadata.view_count,
                    "description": result.metadata.description,
                }
                if result.metadata
                else None,
            }

            async with aiofiles.open(result_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(result_dict, indent=2, ensure_ascii=False))

            logger.debug(
                "Processing result saved",
                video_id=result.video_id,
                path=str(result_path),
            )
            return result_path

        except Exception as e:
            logger.error(
                "Failed to save processing result",
                video_id=result.video_id,
                error=str(e),
            )
            raise StorageError(f"Failed to save processing result: {str(e)}")

    async def load_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """
        Load video metadata from storage.

        Args:
            video_id: Video identifier

        Returns:
            Video metadata if found, None otherwise
        """
        try:
            video_dir = await self.get_video_directory(video_id)
            metadata_path = video_dir / "metadata.json"

            if not metadata_path.exists():
                logger.debug("Metadata file not found", video_id=video_id)
                return None

            async with aiofiles.open(metadata_path, "r", encoding="utf-8") as f:
                content = await f.read()
                metadata_dict = json.loads(content)

            metadata = VideoMetadata(
                video_id=metadata_dict["video_id"],
                title=metadata_dict["title"],
                duration=metadata_dict["duration"],
                url=metadata_dict["url"],
                channel=metadata_dict.get("channel"),
                upload_date=metadata_dict.get("upload_date"),
                view_count=metadata_dict.get("view_count"),
                description=metadata_dict.get("description"),
            )

            logger.debug("Metadata loaded", video_id=video_id)
            return metadata

        except Exception as e:
            logger.error("Failed to load metadata", video_id=video_id, error=str(e))
            return None

    async def load_translation_data(
        self, video_id: str
    ) -> Optional[List[TranslationSegment]]:
        """
        Load translation data from storage.

        Args:
            video_id: Video identifier

        Returns:
            Translation segments if found, None otherwise
        """
        try:
            video_dir = await self.get_video_directory(video_id)
            translation_path = video_dir / "translation.json"

            if not translation_path.exists():
                logger.debug("Translation file not found", video_id=video_id)
                return None

            async with aiofiles.open(translation_path, "r", encoding="utf-8") as f:
                content = await f.read()
                translation_data = json.loads(content)

            # Import here to avoid circular imports
            from ..core.models import TranscriptSegment, TranslationSegment

            segments = []
            for seg_data in translation_data["segments"]:
                original_segment = TranscriptSegment(
                    start_time=seg_data["start_time"],
                    end_time=seg_data["end_time"],
                    text=seg_data["original_text"],
                )

                translation_segment = TranslationSegment(
                    original_segment=original_segment,
                    translated_text=seg_data["translated_text"],
                    language=seg_data.get("language"),
                    audio_path=Path(seg_data["audio_path"])
                    if seg_data.get("audio_path")
                    else None,
                )

                segments.append(translation_segment)

            logger.debug(
                "Translation data loaded",
                video_id=video_id,
                segment_count=len(segments),
            )
            return segments

        except Exception as e:
            logger.error(
                "Failed to load translation data", video_id=video_id, error=str(e)
            )
            return None

    async def get_video_directory(self, video_id: str) -> Path:
        """
        Get the storage directory for a video.

        Args:
            video_id: Video identifier

        Returns:
            Path to video directory
        """
        video_dir = self.base_directory / f"video_{video_id}"
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    async def list_videos(self) -> List[str]:
        """
        Get list of all stored video IDs.

        Returns:
            List of video IDs
        """
        try:
            video_ids = []

            for item in self.base_directory.iterdir():
                if item.is_dir() and item.name.startswith("video_"):
                    video_id = item.name[6:]  # Remove "video_" prefix
                    video_ids.append(video_id)

            logger.debug("Listed videos", count=len(video_ids))
            return video_ids

        except Exception as e:
            logger.error("Failed to list videos", error=str(e))
            raise StorageError(f"Failed to list videos: {str(e)}")

    async def delete_video_data(self, video_id: str) -> bool:
        """
        Delete all data for a video.

        Args:
            video_id: Video identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            video_dir = await self.get_video_directory(video_id)

            if not video_dir.exists():
                logger.debug("Video directory not found", video_id=video_id)
                return False

            # Remove directory and all contents
            import shutil

            shutil.rmtree(video_dir)

            logger.info("Video data deleted", video_id=video_id)
            return True

        except Exception as e:
            logger.error("Failed to delete video data", video_id=video_id, error=str(e))
            return False

    async def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            video_ids = await self.list_videos()
            total_size = 0

            for video_id in video_ids:
                video_dir = await self.get_video_directory(video_id)
                for file_path in video_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

            stats = {
                "video_count": len(video_ids),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "base_directory": str(self.base_directory),
            }

            logger.debug("Storage stats calculated", **stats)
            return stats

        except Exception as e:
            logger.error("Failed to calculate storage stats", error=str(e))
            raise StorageError(f"Failed to calculate storage stats: {str(e)}")

    async def save_timed_transcript(self, video_id: str, timed_transcript) -> Path:
        """Save timed transcript to storage."""
        try:
            video_dir = await self.get_video_directory(video_id)
            transcript_path = video_dir / "timed_transcript.json"
            
            # Convert to dict for JSON serialization
            transcript_dict = {
                "video_id": timed_transcript.video_metadata.video_id,
                "language": timed_transcript.language,
                "source_type": timed_transcript.source_type.value,
                "timing_metadata": {
                    "total_duration": timed_transcript.timing_metadata.total_duration,
                    "segment_count": timed_transcript.timing_metadata.segment_count,
                    "average_segment_duration": timed_transcript.timing_metadata.average_segment_duration,
                    "timing_accuracy": timed_transcript.timing_metadata.timing_accuracy,
                    "has_precise_timing": timed_transcript.timing_metadata.has_precise_timing,
                    "extraction_method": timed_transcript.timing_metadata.extraction_method
                },
                "video_metadata": {
                    "video_id": timed_transcript.video_metadata.video_id,
                    "title": timed_transcript.video_metadata.title,
                    "duration": timed_transcript.video_metadata.duration,
                    "url": timed_transcript.video_metadata.url,
                    "channel": timed_transcript.video_metadata.channel,
                    "upload_date": timed_transcript.video_metadata.upload_date,
                    "view_count": timed_transcript.video_metadata.view_count,
                    "description": timed_transcript.video_metadata.description
                },
                "extraction_quality": timed_transcript.extraction_quality,
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text
                    }
                    for seg in timed_transcript.segments
                ]
            }
            
            async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(transcript_dict, indent=2, ensure_ascii=False))
            
            logger.debug("Timed transcript saved", video_id=video_id, path=str(transcript_path))
            return transcript_path
            
        except Exception as e:
            logger.error("Failed to save timed transcript", video_id=video_id, error=str(e))
            raise StorageError(f"Failed to save timed transcript: {str(e)}")

    async def save_timed_translation(self, video_id: str, timed_translation) -> Path:
        """Save timed translation to storage."""
        try:
            video_dir = await self.get_video_directory(video_id)
            translation_path = video_dir / "timed_translation.json"
            
            # Convert to dict for JSON serialization
            translation_dict = {
                "video_id": timed_translation.original_transcript.video_metadata.video_id,
                "target_language": timed_translation.target_language,
                "created_at": timed_translation.created_at.isoformat() if timed_translation.created_at else None,
                "alignment_config": {
                    "strategy": timed_translation.alignment_config.strategy.value,
                    "parameters": timed_translation.alignment_config.parameters
                },
                "alignment_evaluation": {
                    "timing_accuracy": timed_translation.alignment_evaluation.timing_accuracy,
                    "text_preservation": timed_translation.alignment_evaluation.text_preservation,
                    "boundary_alignment": timed_translation.alignment_evaluation.boundary_alignment,
                    "overall_score": timed_translation.alignment_evaluation.overall_score,
                    "execution_time": timed_translation.alignment_evaluation.execution_time
                },
                "timing_metadata": {
                    "total_duration": timed_translation.timing_metadata.total_duration,
                    "segment_count": timed_translation.timing_metadata.segment_count,
                    "average_segment_duration": timed_translation.timing_metadata.average_segment_duration,
                    "timing_accuracy": timed_translation.timing_metadata.timing_accuracy,
                    "has_precise_timing": timed_translation.timing_metadata.has_precise_timing,
                    "extraction_method": timed_translation.timing_metadata.extraction_method
                },
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "original_text": seg.original_text,
                        "translated_text": seg.translated_text,
                        "confidence": getattr(seg, 'confidence', None)
                    }
                    for seg in timed_translation.segments
                ]
            }
            
            async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(translation_dict, indent=2, ensure_ascii=False))
            
            logger.debug("Timed translation saved", video_id=video_id, path=str(translation_path))
            return translation_path
            
        except Exception as e:
            logger.error("Failed to save timed translation", video_id=video_id, error=str(e))
            raise StorageError(f"Failed to save timed translation: {str(e)}")

    async def load_timed_transcript(self, video_id: str):
        """Load timed transcript from storage."""
        try:
            video_dir = await self.get_video_directory(video_id)
            transcript_path = video_dir / "timed_transcript.json"
            
            if not transcript_path.exists():
                logger.debug("Timed transcript not found", video_id=video_id)
                return None
            
            async with aiofiles.open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.loads(await f.read())
            
            # Import here to avoid circular imports
            from ..core.models import TimedTranscript, TranscriptSegment, VideoMetadata, TimingMetadata, SourceType
            
            segments = [
                TranscriptSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=seg["text"]
                )
                for seg in transcript_data["segments"]
            ]
            
            # Reconstruct metadata objects
            video_metadata = VideoMetadata(**transcript_data["video_metadata"])
            timing_metadata = TimingMetadata(**transcript_data["timing_metadata"])
            source_type = SourceType(transcript_data["source_type"])
            
            timed_transcript = TimedTranscript(
                segments=segments,
                source_type=source_type,
                timing_metadata=timing_metadata,
                video_metadata=video_metadata,
                language=transcript_data["language"],
                extraction_quality=transcript_data.get("extraction_quality")
            )
            
            logger.debug("Timed transcript loaded", video_id=video_id)
            return timed_transcript
            
        except Exception as e:
            logger.error("Failed to load timed transcript", video_id=video_id, error=str(e))
            return None

    async def load_timed_translation(self, video_id: str):
        """Load timed translation from storage."""
        try:
            video_dir = await self.get_video_directory(video_id)
            translation_path = video_dir / "timed_translation.json"
            
            if not translation_path.exists():
                logger.debug("Timed translation not found", video_id=video_id)
                return None
            
            async with aiofiles.open(translation_path, "r", encoding="utf-8") as f:
                translation_data = json.loads(await f.read())
            
            # Import here to avoid circular imports
            from ..core.models import (
                TimedTranslation, 
                TimedTranslationSegment, 
                AlignmentConfig, 
                AlignmentEvaluation, 
                AlignmentStrategy,
                TimingMetadata
            )
            from datetime import datetime
            
            segments = [
                TimedTranslationSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    original_text=seg["original_text"],
                    translated_text=seg["translated_text"],
                    confidence=seg.get("confidence")
                )
                for seg in translation_data["segments"]
            ]
            
            # Load the original transcript from storage
            original_transcript = await self.load_timed_transcript(video_id)
            if not original_transcript:
                logger.warning("Original transcript not found for timed translation", video_id=video_id)
                return None
            
            # Reconstruct objects
            alignment_config = AlignmentConfig(
                strategy=AlignmentStrategy(translation_data["alignment_config"]["strategy"]),
                parameters=translation_data["alignment_config"]["parameters"]
            )
            
            alignment_evaluation = AlignmentEvaluation(**translation_data["alignment_evaluation"])
            timing_metadata = TimingMetadata(**translation_data["timing_metadata"])
            
            created_at = None
            if translation_data.get("created_at"):
                created_at = datetime.fromisoformat(translation_data["created_at"])
            
            timed_translation = TimedTranslation(
                segments=segments,
                original_transcript=original_transcript,
                target_language=translation_data["target_language"],
                alignment_config=alignment_config,
                alignment_evaluation=alignment_evaluation,
                timing_metadata=timing_metadata,
                created_at=created_at
            )
            
            logger.debug("Timed translation loaded", video_id=video_id)
            return timed_translation
            
        except Exception as e:
            logger.error("Failed to load timed translation", video_id=video_id, error=str(e))
            return None
