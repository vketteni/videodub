"""Pytest configuration and fixtures."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

import pytest

from videodub import (
    PipelineConfig,
    VideoMetadata,
    TranscriptSegment,
    TranslationSegment,
    TTSEngine,
    FileStorageService,
    OpenAITranslationService,
    OpenAITTSService,
    YouTubeScrapingService,
    TranslationPipeline,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_video_metadata() -> VideoMetadata:
    """Sample video metadata for testing."""
    return VideoMetadata(
        video_id="test_video_123",
        title="Test Video Title",
        duration=120.0,
        url="https://www.youtube.com/watch?v=test123",
        channel="Test Channel",
        upload_date="2024-01-01",
        view_count=1000,
        description="Test video description"
    )


@pytest.fixture
def sample_transcript_segments() -> list[TranscriptSegment]:
    """Sample transcript segments for testing."""
    return [
        TranscriptSegment(
            start_time=0.0,
            end_time=3.0,
            text="Hello, this is a test video."
        ),
        TranscriptSegment(
            start_time=3.0,
            end_time=6.0,
            text="We are testing the translation pipeline."
        ),
        TranscriptSegment(
            start_time=6.0,
            end_time=9.0,
            text="Thank you for watching!"
        )
    ]


@pytest.fixture
def sample_translation_segments(sample_transcript_segments) -> list[TranslationSegment]:
    """Sample translation segments for testing."""
    return [
        TranslationSegment(
            original_segment=sample_transcript_segments[0],
            translated_text="Hola, este es un video de prueba.",
            language="es"
        ),
        TranslationSegment(
            original_segment=sample_transcript_segments[1],
            translated_text="Estamos probando el pipeline de traducción.",
            language="es"
        ),
        TranslationSegment(
            original_segment=sample_transcript_segments[2],
            translated_text="¡Gracias por mirar!",
            language="es"
        )
    ]


@pytest.fixture
def pipeline_config(temp_dir: Path) -> PipelineConfig:
    """Pipeline configuration for testing."""
    return PipelineConfig(
        output_directory=temp_dir / "output",
        target_language="es",
        tts_engine=TTSEngine.SYSTEM,  # Use system TTS for tests
        max_concurrent_requests=2,
        request_timeout=10,
        temp_directory=temp_dir / "temp",
        audio_format="wav"
    )


@pytest.fixture
def storage_service(temp_dir: Path) -> FileStorageService:
    """File storage service for testing."""
    return FileStorageService(temp_dir / "storage")


@pytest.fixture
def mock_video_service() -> Mock:
    """Mock video scraping service."""
    service = Mock(spec=YouTubeScrapingService)
    
    # Configure async methods
    service.scrape_video = AsyncMock()
    service.scrape_audio_only = AsyncMock()
    service.get_transcript = AsyncMock()
    
    return service


@pytest.fixture
def mock_translation_service() -> Mock:
    """Mock translation service."""
    service = Mock(spec=OpenAITranslationService)
    
    # Configure async methods
    service.translate_text = AsyncMock()
    service.translate_segments = AsyncMock()
    service.translate_batch = AsyncMock()
    
    return service


@pytest.fixture
def mock_tts_service() -> Mock:
    """Mock TTS service."""
    service = Mock(spec=OpenAITTSService)
    
    # Configure async methods
    service.generate_audio = AsyncMock()
    service.generate_batch_audio = AsyncMock()
    service.get_supported_languages = Mock(return_value=["en", "es", "fr"])
    service.get_supported_voices = Mock(return_value=["alloy", "echo"])
    
    return service


@pytest.fixture
def mock_pipeline(
    mock_video_service: Mock,
    mock_translation_service: Mock,
    mock_tts_service: Mock,
    storage_service: FileStorageService,
    pipeline_config: PipelineConfig
) -> TranslationPipeline:
    """Mock translation pipeline with all services mocked."""
    return TranslationPipeline(
        video_service=mock_video_service,
        translation_service=mock_translation_service,
        tts_service=mock_tts_service,
        storage_service=storage_service,
        config=pipeline_config
    )


@pytest.fixture
def api_key() -> str:
    """Mock API key for testing."""
    return "sk-test-api-key-for-testing"


@pytest.fixture
def sample_video_url() -> str:
    """Sample video URL for testing."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# Async fixtures
@pytest.fixture
async def async_storage_with_data(
    storage_service: FileStorageService,
    sample_video_metadata: VideoMetadata,
    sample_translation_segments: list[TranslationSegment]
) -> FileStorageService:
    """Storage service with sample data."""
    # Save sample data
    await storage_service.save_metadata("test_video_123", sample_video_metadata)
    await storage_service.save_translation_data("test_video_123", sample_translation_segments)
    
    return storage_service


# Marks for different test types
def pytest_configure(config):
    """Configure pytest with custom marks."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API"
    )


# Skip tests that require API keys if not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip API tests if no keys available."""
    import os
    
    skip_api = pytest.mark.skip(reason="API key not available")
    
    for item in items:
        if "requires_api" in item.keywords:
            if not os.getenv("OPENAI_API_KEY"):
                item.add_marker(skip_api)