"""Tests for data extraction service."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.videodub.core.exceptions import DataExtractionError
from src.videodub.core.models import SourceType, TimedTranscript, TranscriptSegment
from src.videodub.services.data_extraction import YouTubeDataExtractionService


class TestYouTubeDataExtractionService:
    """Test suite for YouTubeDataExtractionService."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return tmp_path / "test_output"

    @pytest.fixture
    def mock_scraper_result(self):
        """Create a mock scraper result."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.error = None
        mock_result.video_id = "test_video_id"

        # Mock metadata
        mock_metadata = Mock()
        mock_metadata.video_id = "test_video_id"
        mock_metadata.title = "Test Video"
        mock_metadata.duration = 120.0
        mock_metadata.channel = "Test Channel"
        mock_metadata.upload_date = "2023-01-01"
        mock_metadata.view_count = 1000
        mock_metadata.description = "Test description"

        # Mock transcript
        mock_transcript_entry = Mock()
        mock_transcript_entry.start = 0.0
        mock_transcript_entry.end = 3.0
        mock_transcript_entry.text = "Test transcript text"
        mock_metadata.transcript = [mock_transcript_entry]

        mock_result.metadata = mock_metadata
        return mock_result

    @pytest.fixture
    def service(self, output_dir):
        """Create a YouTubeDataExtractionService instance."""
        with patch("video_scraper.VideoScraperAPI"), patch(
            "video_scraper.scrape_audio_only"
        ):
            return YouTubeDataExtractionService(output_dir)

    def test_supports_source_youtube_url(self, service):
        """Test that service supports YouTube URLs."""
        assert service.supports_source("https://www.youtube.com/watch?v=test") is True
        assert service.supports_source("https://youtu.be/test") is True
        assert service.supports_source("https://example.com/video") is False
        assert service.supports_source("/local/file.mp4") is False

    @pytest.mark.asyncio
    async def test_extract_from_url_success(self, service, mock_scraper_result):
        """Test successful data extraction from URL."""
        test_url = "https://www.youtube.com/watch?v=test"

        # Mock the scraper method
        service._scrape_video_sync = Mock(return_value=mock_scraper_result)

        # Mock validation
        with patch("src.videodub.services.data_extraction.validate_video_url"):
            result = await service.extract_from_url(test_url)

        # Verify result
        assert isinstance(result, TimedTranscript)
        assert result.source_type == SourceType.YOUTUBE
        assert len(result.segments) == 1
        assert result.segments[0].text == "Test transcript text"
        assert result.video_metadata.title == "Test Video"
        assert result.timing_metadata.segment_count == 1
        assert result.timing_metadata.total_duration == 3.0

    @pytest.mark.asyncio
    async def test_extract_from_url_scraper_failure(self, service):
        """Test handling of scraper failure."""
        test_url = "https://www.youtube.com/watch?v=test"

        # Mock failed scraper result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Scraping failed"

        service._scrape_video_sync = Mock(return_value=mock_result)

        with patch("src.videodub.services.data_extraction.validate_video_url"):
            with pytest.raises(DataExtractionError) as exc_info:
                await service.extract_from_url(test_url)

            assert "Failed to scrape video" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_from_file_not_implemented(self, service, tmp_path):
        """Test that file extraction raises NotImplementedError."""
        # Create a temporary file that exists
        test_file = tmp_path / "video.mp4"
        test_file.write_text("fake video content")

        with pytest.raises(NotImplementedError) as exc_info:
            await service.extract_from_file(test_file)

        assert "Local file transcription not yet implemented" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_from_file_not_found(self, service):
        """Test handling of non-existent file."""
        test_file = Path("/nonexistent/video.mp4")

        with pytest.raises(DataExtractionError) as exc_info:
            await service.extract_from_file(test_file)

        assert "Video file not found" in str(exc_info.value)

    def test_create_timing_metadata(self, service):
        """Test timing metadata creation."""
        segments = [
            TranscriptSegment(0.0, 3.0, "First segment"),
            TranscriptSegment(3.0, 6.0, "Second segment"),
            TranscriptSegment(6.0, 9.0, "Third segment"),
        ]

        timing_metadata = service._create_timing_metadata(segments)

        assert timing_metadata.total_duration == 9.0
        assert timing_metadata.segment_count == 3
        assert timing_metadata.average_segment_duration == 3.0
        assert timing_metadata.has_precise_timing is True
        assert timing_metadata.extraction_method == "youtube_api"

    def test_create_timing_metadata_empty(self, service):
        """Test timing metadata creation with empty segments."""
        segments = []

        timing_metadata = service._create_timing_metadata(segments)

        assert timing_metadata.total_duration == 0.0
        assert timing_metadata.segment_count == 0
        assert timing_metadata.average_segment_duration == 0.0
        assert timing_metadata.has_precise_timing is False

    def test_calculate_timing_accuracy_perfect(self, service):
        """Test timing accuracy calculation for perfect timing."""
        segments = [
            TranscriptSegment(0.0, 3.0, "First"),
            TranscriptSegment(3.0, 6.0, "Second"),
            TranscriptSegment(6.0, 9.0, "Third"),
        ]

        accuracy = service._calculate_timing_accuracy(segments)

        assert accuracy == 1.0  # Perfect timing

    def test_calculate_timing_accuracy_with_gaps(self, service):
        """Test timing accuracy calculation with gaps."""
        segments = [
            TranscriptSegment(0.0, 3.0, "First"),
            TranscriptSegment(4.0, 7.0, "Second"),  # 1 second gap
            TranscriptSegment(8.0, 11.0, "Third"),  # 1 second gap
        ]

        accuracy = service._calculate_timing_accuracy(segments)

        assert accuracy < 1.0  # Should be penalized for gaps

    def test_calculate_extraction_quality(self, service):
        """Test extraction quality calculation."""
        segments = [
            TranscriptSegment(
                0.0, 3.0, "This is a good length segment with plenty of text"
            ),
            TranscriptSegment(3.0, 6.0, "Another good segment"),
            TranscriptSegment(6.0, 9.0, "Short"),
        ]

        quality = service._calculate_extraction_quality(segments)

        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be reasonable quality

    def test_extract_video_id_from_url(self, service):
        """Test video ID extraction from various URL formats."""
        # Standard YouTube URL
        assert (
            service._extract_video_id_from_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            == "dQw4w9WgXcQ"
        )

        # Short YouTube URL
        assert (
            service._extract_video_id_from_url("https://youtu.be/dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

        # URL with additional parameters
        assert (
            service._extract_video_id_from_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"
            )
            == "dQw4w9WgXcQ"
        )

        # Non-YouTube URL should return hash-based ID
        result = service._extract_video_id_from_url("https://example.com/video")
        assert result.startswith("video_")
        assert len(result) == 10  # "video_" + 4 digits
