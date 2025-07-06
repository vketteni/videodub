"""Data extraction service implementation."""

import asyncio
from pathlib import Path
from typing import Any, Optional

import structlog

from ..config.validation import validate_video_url
from ..core.exceptions import DataExtractionError
from ..core.interfaces import DataExtractionService
from ..core.models import (
    SourceType,
    TimedTranscript,
    TimingMetadata,
    TranscriptSegment,
    VideoMetadata,
)

logger = structlog.get_logger(__name__)


class YouTubeDataExtractionService(DataExtractionService):
    """YouTube data extraction service using video_scraper package."""

    def __init__(self, output_dir: Path, audio_format: str = "mp3"):
        """
        Initialize the data extraction service.

        Args:
            output_dir: Directory to save downloaded content
            audio_format: Audio format for extraction
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_format = audio_format
        self._setup_scraper()

    def _setup_scraper(self) -> None:
        """Set up the underlying video scraper."""
        try:
            from video_scraper import VideoScraperAPI, scrape_audio_only

            self.scraper = VideoScraperAPI(
                output_dir=str(self.output_dir), extract_audio=True
            )
            self.audio_scraper = scrape_audio_only
            logger.info("Video scraper initialized", output_dir=str(self.output_dir))
        except ImportError as e:
            raise DataExtractionError(
                "video_scraper package not available", details={"error": str(e)}
            )

    async def extract_from_url(self, url: str) -> TimedTranscript:
        """
        Extract transcript and timing data from video URL.

        Args:
            url: Video URL to extract from

        Returns:
            TimedTranscript with extracted data

        Raises:
            DataExtractionError: If extraction fails
        """
        validate_video_url(url)

        logger.info("Starting data extraction from URL", url=url)

        try:
            # Use existing scraper logic
            metadata, transcript_segments = await self._scrape_video_data(url)

            # Create timing metadata
            timing_metadata = self._create_timing_metadata(transcript_segments)

            # Create timed transcript
            timed_transcript = TimedTranscript(
                segments=transcript_segments,
                source_type=SourceType.YOUTUBE,
                timing_metadata=timing_metadata,
                video_metadata=metadata,
                language=self._detect_language(transcript_segments),
                extraction_quality=self._calculate_extraction_quality(
                    transcript_segments
                ),
            )

            logger.info(
                "Data extraction completed",
                video_id=metadata.video_id,
                segments=len(transcript_segments),
                duration=timing_metadata.total_duration,
                quality=timed_transcript.extraction_quality,
            )

            return timed_transcript

        except Exception as e:
            logger.error("Data extraction failed", url=url, error=str(e))
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(
                f"Unexpected error during data extraction: {str(e)}",
                details={"url": url},
            )

    async def extract_from_file(self, file_path: Path) -> TimedTranscript:
        """
        Extract transcript and timing data from local video file.

        Args:
            file_path: Path to video file

        Returns:
            TimedTranscript with extracted data

        Raises:
            DataExtractionError: If extraction fails
        """
        if not file_path.exists():
            raise DataExtractionError(f"Video file not found: {file_path}")

        logger.info("Starting data extraction from file", file_path=str(file_path))

        # For now, raise NotImplementedError as this requires speech-to-text
        raise NotImplementedError(
            "Local file transcription not yet implemented. "
            "This requires speech-to-text capabilities."
        )

    def supports_source(self, source: str) -> bool:
        """
        Check if this service supports the given source.

        Args:
            source: URL or file path to check

        Returns:
            True if source is supported
        """
        # Check if it's a YouTube URL
        if isinstance(source, str):
            return "youtube.com" in source or "youtu.be" in source
        return False

    async def _scrape_video_data(
        self, url: str
    ) -> tuple[VideoMetadata, list[TranscriptSegment]]:
        """Scrape video data using existing scraper logic."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._scrape_video_sync, url)

            if not result.success:
                raise DataExtractionError(
                    f"Failed to scrape video: {result.error}",
                    details={"url": url, "error": result.error},
                )

            # Extract metadata
            metadata = self._extract_metadata(url, result)

            # Extract transcript
            transcript_segments = self._extract_transcript(result)

            return metadata, transcript_segments

        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(
                f"Failed to scrape video data: {str(e)}", details={"url": url}
            )

    def _scrape_video_sync(self, url: str) -> Any:
        """Synchronous video scraping for thread pool execution."""
        return self.scraper.scrape(url)

    def _extract_metadata(self, url: str, result: Any) -> VideoMetadata:
        """Extract metadata from scraper result."""
        return VideoMetadata(
            video_id=getattr(result, "video_id", None)
            or getattr(result.metadata, "video_id", None)
            or self._extract_video_id_from_url(url),
            title=result.metadata.title,
            duration=result.metadata.duration,
            url=url,
            channel=getattr(result.metadata, "channel", None),
            upload_date=getattr(result.metadata, "upload_date", None),
            view_count=getattr(result.metadata, "view_count", None),
            description=getattr(result.metadata, "description", None),
        )

    def _extract_transcript(self, result: Any) -> list[TranscriptSegment]:
        """Extract transcript segments from scraper result."""
        transcript_segments = []

        # Check if transcript exists in metadata
        if hasattr(result.metadata, "transcript") and result.metadata.transcript:
            for entry in result.metadata.transcript:
                try:
                    segment = TranscriptSegment(
                        start_time=float(entry.start),
                        end_time=float(getattr(entry, "end", entry.start + 3)),
                        text=str(entry.text).strip(),
                    )
                    transcript_segments.append(segment)
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        "Skipping invalid transcript entry",
                        entry=str(entry),
                        error=str(e),
                    )
                    continue

        # Try to load from metadata file if transcript not found
        if not transcript_segments:
            transcript_segments = self._try_load_transcript_from_file(result)

        return transcript_segments

    def _try_load_transcript_from_file(self, result: Any) -> list[TranscriptSegment]:
        """Try to load transcript from metadata file."""
        transcript_segments = []

        try:
            video_id = getattr(result.metadata, "video_id", "unknown")
            metadata_file = self.output_dir / video_id / "metadata.json"

            if metadata_file.exists():
                import json

                with open(metadata_file) as f:
                    metadata = json.load(f)

                if "transcript" in metadata:
                    for entry in metadata["transcript"]:
                        try:
                            segment = TranscriptSegment(
                                start_time=float(entry["start"]),
                                end_time=float(entry.get("end", entry["start"] + 3)),
                                text=str(entry["text"]).strip(),
                            )
                            transcript_segments.append(segment)
                        except (ValueError, KeyError) as e:
                            logger.warning(
                                "Skipping invalid transcript entry from file",
                                entry=entry,
                                error=str(e),
                            )
                            continue

        except Exception as e:
            logger.warning("Could not load transcript from file", error=str(e))

        return transcript_segments

    def _extract_video_id_from_url(self, url: str) -> str:
        """Extract video ID from URL for naming purposes."""
        try:
            if "youtube.com" in url or "youtu.be" in url:
                if "v=" in url:
                    return url.split("v=")[1].split("&")[0]
                elif "youtu.be/" in url:
                    return url.split("youtu.be/")[1].split("?")[0]

            # Fallback to hash-based ID
            return f"video_{hash(url) % 10000:04d}"

        except Exception:
            return f"video_{hash(url) % 10000:04d}"

    def _create_timing_metadata(
        self, segments: list[TranscriptSegment]
    ) -> TimingMetadata:
        """Create timing metadata from transcript segments."""
        if not segments:
            return TimingMetadata(
                total_duration=0.0,
                segment_count=0,
                average_segment_duration=0.0,
                has_precise_timing=False,
            )

        total_duration = max(s.end_time for s in segments)
        segment_count = len(segments)
        total_segment_duration = sum(s.end_time - s.start_time for s in segments)
        average_segment_duration = total_segment_duration / segment_count

        # Calculate timing accuracy based on segment overlap and gaps
        timing_accuracy = self._calculate_timing_accuracy(segments)

        return TimingMetadata(
            total_duration=total_duration,
            segment_count=segment_count,
            average_segment_duration=average_segment_duration,
            timing_accuracy=timing_accuracy,
            has_precise_timing=timing_accuracy >= 0.8,  # Threshold for "precise"
            extraction_method="youtube_api",
        )

    def _calculate_timing_accuracy(self, segments: list[TranscriptSegment]) -> float:
        """Calculate timing accuracy score based on segment continuity."""
        if len(segments) < 2:
            return 1.0

        gap_penalty = 0.0
        overlap_penalty = 0.0

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # Calculate gap or overlap
            gap = next_seg.start_time - current.end_time

            if gap > 0.5:  # Gap > 0.5 seconds
                gap_penalty += min(gap, 2.0) / 2.0  # Cap penalty at 1.0
            elif gap < -0.1:  # Overlap > 0.1 seconds
                overlap_penalty += min(abs(gap), 1.0)

        # Convert penalties to accuracy score (0.0 to 1.0)
        total_penalty = (gap_penalty + overlap_penalty) / len(segments)
        accuracy = max(0.0, 1.0 - total_penalty)

        return accuracy

    def _detect_language(self, segments: list[TranscriptSegment]) -> Optional[str]:
        """Detect language from transcript segments."""
        # For now, return None - language detection can be added later
        # This could use langdetect or similar library
        return None

    def _calculate_extraction_quality(self, segments: list[TranscriptSegment]) -> float:
        """Calculate overall extraction quality score."""
        if not segments:
            return 0.0

        # Quality based on segment characteristics
        quality_score = 0.0

        # Text completeness (segments with reasonable text length)
        text_scores = []
        for segment in segments:
            text_length = len(segment.text.strip())
            if text_length == 0:
                text_scores.append(0.0)
            elif text_length < 10:
                text_scores.append(0.3)
            elif text_length < 50:
                text_scores.append(0.7)
            else:
                text_scores.append(1.0)

        text_quality = sum(text_scores) / len(text_scores)

        # Timing quality (already calculated)
        timing_quality = self._calculate_timing_accuracy(segments)

        # Segment density (reasonable number of segments for duration)
        if segments:
            total_duration = max(s.end_time for s in segments)
            expected_segments = total_duration / 3.0  # Expect ~3 second segments
            density_ratio = len(segments) / max(expected_segments, 1.0)
            density_quality = 1.0 - min(abs(1.0 - density_ratio), 1.0)
        else:
            density_quality = 0.0

        # Weighted average
        quality_score = (
            text_quality * 0.5 + timing_quality * 0.3 + density_quality * 0.2
        )

        return round(quality_score, 3)
