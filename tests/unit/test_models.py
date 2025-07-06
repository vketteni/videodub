"""Unit tests for core models."""

from datetime import datetime
from pathlib import Path

import pytest

from videodub.core.exceptions import ValidationError
from videodub.core.models import (
    PipelineConfig,
    ProcessingResult,
    ProcessingStatus,
    TranscriptSegment,
    TranslationSegment,
    TTSEngine,
    VideoMetadata,
)


class TestVideoMetadata:
    """Test VideoMetadata model."""

    def test_create_video_metadata(self):
        """Test creating video metadata."""
        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
            duration=120.0,
            url="https://youtube.com/watch?v=test123",
        )

        assert metadata.video_id == "test123"
        assert metadata.title == "Test Video"
        assert metadata.duration == 120.0
        assert metadata.url == "https://youtube.com/watch?v=test123"
        assert metadata.channel is None
        assert metadata.upload_date is None
        assert metadata.view_count is None
        assert metadata.description is None

    def test_video_metadata_with_optional_fields(self):
        """Test video metadata with all optional fields."""
        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
            duration=120.0,
            url="https://youtube.com/watch?v=test123",
            channel="Test Channel",
            upload_date="2024-01-01",
            view_count=1000,
            description="Test description",
        )

        assert metadata.channel == "Test Channel"
        assert metadata.upload_date == "2024-01-01"
        assert metadata.view_count == 1000
        assert metadata.description == "Test description"


class TestTranscriptSegment:
    """Test TranscriptSegment model."""

    def test_create_transcript_segment(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(start_time=0.0, end_time=3.0, text="Hello world")

        assert segment.start_time == 0.0
        assert segment.end_time == 3.0
        assert segment.text == "Hello world"

    def test_transcript_segment_validation(self):
        """Test transcript segment validation."""
        # Negative start time
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            TranscriptSegment(start_time=-1.0, end_time=3.0, text="Hello")

        # End time not greater than start time
        with pytest.raises(
            ValueError, match="End time must be greater than start time"
        ):
            TranscriptSegment(start_time=3.0, end_time=3.0, text="Hello")

        with pytest.raises(
            ValueError, match="End time must be greater than start time"
        ):
            TranscriptSegment(start_time=3.0, end_time=2.0, text="Hello")

        # Empty text
        with pytest.raises(ValueError, match="Text cannot be empty"):
            TranscriptSegment(start_time=0.0, end_time=3.0, text="")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            TranscriptSegment(start_time=0.0, end_time=3.0, text="   ")


class TestTranslationSegment:
    """Test TranslationSegment model."""

    def test_create_translation_segment(self):
        """Test creating a translation segment."""
        original = TranscriptSegment(start_time=0.0, end_time=3.0, text="Hello world")

        translation = TranslationSegment(
            original_segment=original, translated_text="Hola mundo", language="es"
        )

        assert translation.original_segment == original
        assert translation.translated_text == "Hola mundo"
        assert translation.language == "es"
        assert translation.audio_path is None

    def test_translation_segment_with_audio_path(self):
        """Test translation segment with audio path."""
        original = TranscriptSegment(start_time=0.0, end_time=3.0, text="Hello world")

        audio_path = Path("/test/audio.wav")
        translation = TranslationSegment(
            original_segment=original,
            translated_text="Hola mundo",
            audio_path=audio_path,
        )

        assert translation.audio_path == audio_path

    def test_translation_segment_validation(self):
        """Test translation segment validation."""
        original = TranscriptSegment(start_time=0.0, end_time=3.0, text="Hello world")

        # Empty translated text
        with pytest.raises(ValueError, match="Translated text cannot be empty"):
            TranslationSegment(original_segment=original, translated_text="")

        with pytest.raises(ValueError, match="Translated text cannot be empty"):
            TranslationSegment(original_segment=original, translated_text="   ")


class TestProcessingResult:
    """Test ProcessingResult model."""

    def test_create_processing_result(self):
        """Test creating a processing result."""
        result = ProcessingResult(
            video_id="test123", status=ProcessingStatus.PROCESSING
        )

        assert result.video_id == "test123"
        assert result.status == ProcessingStatus.PROCESSING
        assert isinstance(result.started_at, datetime)
        assert result.completed_at is None
        assert result.metadata is None
        assert result.files == {}
        assert result.errors == []
        assert result.target_language is None
        assert result.tts_engine is None

    def test_add_error(self):
        """Test adding errors to processing result."""
        result = ProcessingResult(
            video_id="test123", status=ProcessingStatus.PROCESSING
        )

        result.add_error("Test error")

        assert result.errors == ["Test error"]
        assert result.status == ProcessingStatus.FAILED

    def test_mark_completed(self):
        """Test marking result as completed."""
        result = ProcessingResult(
            video_id="test123", status=ProcessingStatus.PROCESSING
        )

        result.mark_completed()

        assert result.status == ProcessingStatus.COMPLETED
        assert isinstance(result.completed_at, datetime)

    def test_mark_failed(self):
        """Test marking result as failed."""
        result = ProcessingResult(
            video_id="test123", status=ProcessingStatus.PROCESSING
        )

        result.mark_failed("Something went wrong")

        assert result.status == ProcessingStatus.FAILED
        assert isinstance(result.completed_at, datetime)
        assert "Something went wrong" in result.errors


class TestPipelineConfig:
    """Test PipelineConfig model."""

    def test_create_pipeline_config(self, temp_dir):
        """Test creating pipeline configuration."""
        config = PipelineConfig(output_directory=temp_dir / "output")

        assert config.output_directory == temp_dir / "output"
        assert config.target_language == "es"
        assert config.tts_engine == TTSEngine.OPENAI
        assert config.max_concurrent_requests == 5
        assert config.request_timeout == 30
        assert config.temp_directory is None
        assert config.audio_format == "wav"
        assert config.video_quality == "720p"
        assert config.extract_audio is True
        assert config.extract_transcript is True

    def test_pipeline_config_path_conversion(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        config = PipelineConfig(
            output_directory=str(temp_dir / "output"),
            temp_directory=str(temp_dir / "temp"),
        )

        assert isinstance(config.output_directory, Path)
        assert isinstance(config.temp_directory, Path)
        assert config.output_directory == temp_dir / "output"
        assert config.temp_directory == temp_dir / "temp"

    def test_pipeline_config_creates_directories(self, temp_dir):
        """Test that configuration creates directories."""
        output_dir = temp_dir / "output"
        temp_dir_path = temp_dir / "temp"

        assert not output_dir.exists()
        assert not temp_dir_path.exists()

        config = PipelineConfig(
            output_directory=output_dir, temp_directory=temp_dir_path
        )

        assert output_dir.exists()
        assert temp_dir_path.exists()


