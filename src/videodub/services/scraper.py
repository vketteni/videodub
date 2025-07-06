"""Video scraping service implementation."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..config.validation import validate_url, validate_video_url
from ..core.exceptions import VideoScrapingError
from ..core.interfaces import VideoScrapingService
from ..core.models import TranscriptSegment, VideoMetadata

logger = structlog.get_logger(__name__)


class YouTubeScrapingService(VideoScrapingService):
    """YouTube video scraping service using video_scraper package."""

    def __init__(self, output_dir: Path, audio_format: str = "mp3"):
        """
        Initialize the video scraping service.

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
            raise VideoScrapingError(
                "video_scraper package not available", details={"error": str(e)}
            )

    async def scrape_video(
        self, url: str
    ) -> Tuple[VideoMetadata, List[TranscriptSegment]]:
        """
        Scrape video and extract metadata and transcript.

        Args:
            url: Video URL to scrape

        Returns:
            Tuple of video metadata and transcript segments

        Raises:
            VideoScrapingError: If scraping fails
        """
        validate_video_url(url)

        logger.info("Starting video scrape", url=url)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._scrape_video_sync, url)

            if not result.success:
                raise VideoScrapingError(
                    f"Failed to scrape video: {result.error}",
                    details={"url": url, "error": result.error},
                )

            # Extract metadata
            metadata = self._extract_metadata(url, result)

            # Extract transcript
            transcript_segments = self._extract_transcript(result)

            logger.info(
                "Video scrape completed",
                video_id=metadata.video_id,
                duration=metadata.duration,
                transcript_segments=len(transcript_segments),
            )

            return metadata, transcript_segments

        except Exception as e:
            logger.error("Video scraping failed", url=url, error=str(e))
            if isinstance(e, VideoScrapingError):
                raise
            raise VideoScrapingError(
                f"Unexpected error during video scraping: {str(e)}",
                details={"url": url},
            )

    async def scrape_audio_only(self, url: str) -> Tuple[VideoMetadata, Path]:
        """
        Scrape only audio from video.

        Args:
            url: Video URL to scrape

        Returns:
            Tuple of video metadata and audio file path

        Raises:
            VideoScrapingError: If scraping fails
        """
        validate_video_url(url)

        logger.info("Starting audio-only scrape", url=url)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.audio_scraper, url)

            if not result.success:
                raise VideoScrapingError(
                    f"Failed to scrape audio: {result.error}",
                    details={"url": url, "error": result.error},
                )

            # Extract metadata
            metadata = VideoMetadata(
                video_id=getattr(result, "video_id", None)
                or self._extract_video_id_from_url(url),
                title=result.metadata.title,
                duration=result.metadata.duration,
                url=url,
                channel=getattr(result.metadata, "channel", None),
                upload_date=getattr(result.metadata, "upload_date", None),
                view_count=getattr(result.metadata, "view_count", None),
                description=getattr(result.metadata, "description", None),
            )

            # Get audio file path
            audio_path = Path(
                getattr(result, "file_path", None)
                or getattr(result, "audio_path", None)
            )

            if not audio_path or not audio_path.exists():
                raise VideoScrapingError(
                    "Audio file not found after scraping",
                    details={"url": url, "expected_path": str(audio_path)},
                )

            logger.info(
                "Audio scrape completed",
                video_id=metadata.video_id,
                duration=metadata.duration,
                audio_path=str(audio_path),
            )

            return metadata, audio_path

        except Exception as e:
            logger.error("Audio scraping failed", url=url, error=str(e))
            if isinstance(e, VideoScrapingError):
                raise
            raise VideoScrapingError(
                f"Unexpected error during audio scraping: {str(e)}",
                details={"url": url},
            )

    async def get_transcript(self, url: str) -> List[TranscriptSegment]:
        """
        Extract transcript from video.

        Args:
            url: Video URL

        Returns:
            List of transcript segments

        Raises:
            VideoScrapingError: If transcript extraction fails
        """
        validate_video_url(url)

        logger.info("Extracting transcript", url=url)

        try:
            # First scrape video to get transcript
            _, transcript_segments = await self.scrape_video(url)

            logger.info(
                "Transcript extraction completed",
                url=url,
                segments=len(transcript_segments),
            )

            return transcript_segments

        except Exception as e:
            logger.error("Transcript extraction failed", url=url, error=str(e))
            if isinstance(e, VideoScrapingError):
                raise
            raise VideoScrapingError(
                f"Failed to extract transcript: {str(e)}", details={"url": url}
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

    def _extract_transcript(self, result: Any) -> List[TranscriptSegment]:
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

    def _try_load_transcript_from_file(self, result: Any) -> List[TranscriptSegment]:
        """Try to load transcript from metadata file."""
        transcript_segments = []

        try:
            video_id = getattr(result.metadata, "video_id", "unknown")
            metadata_file = self.output_dir / video_id / "metadata.json"

            if metadata_file.exists():
                import json

                with open(metadata_file, "r") as f:
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
