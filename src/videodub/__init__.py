"""YouTube Translation Pipeline - A comprehensive AI-powered video translation system."""

__version__ = "1.0.0"
__author__ = "Vincent"
__email__ = "vincent@example.com"

# Core imports
from .core.models import (
    VideoMetadata,
    TranscriptSegment,
    ProcessedSegment,
    TranslationSegment,
    ProcessingResult,
    ProcessingStatus,
    ProcessingMode,
    TTSEngine,
    PipelineConfig,
    TranslationJob,
    AudioGenerationJob,
)

from .core.pipeline import TranslationPipeline

from .core.exceptions import (
    PipelineError,
    VideoScrapingError,
    TranslationError,
    TTSError,
    ConfigurationError,
    StorageError,
    AudioProcessingError,
    ValidationError,
    APIError,
    RateLimitError,
    AuthenticationError,
    QuotaExceededError,
)

# Configuration
from .config.settings import (
    PipelineSettings,
    LanguageConfig,
    TTSEngineConfig,
    get_settings,
    load_settings,
    validate_configuration,
)

# Services
from .services.scraper import YouTubeScrapingService
from .services.transcript import HybridTranscriptProcessingService, ProcessingConfig
from .services.translator import OpenAITranslationService, FallbackTranslationService
from .services.tts import OpenAITTSService, SystemTTSService, create_tts_service
from .services.storage import FileStorageService
from .services.audio import create_audio_processing_service
from .services.video import FFmpegVideoProcessingService

# Utilities
from .utils.logging import configure_logging, get_logger, setup_pipeline_logging

# Factory functions for easy initialization
def create_pipeline(
    output_directory: str = "./output",
    target_language: str = "es",
    tts_engine: TTSEngine = TTSEngine.OPENAI,
    openai_api_key: str = None,
    translation_model: str = "gpt-4.1-nano",
    tts_model: str = "tts-1",
    **kwargs
) -> TranslationPipeline:
    """
    Create a configured translation pipeline.
    
    Args:
        output_directory: Directory for output files
        target_language: Target language code
        tts_engine: TTS engine to use
        openai_api_key: OpenAI API key (required for OpenAI services)
        translation_model: OpenAI model for translation (e.g., gpt-4.1-nano, gpt-4)
        tts_model: OpenAI model for TTS (e.g., tts-1, tts-1-hd)
        **kwargs: Additional configuration options
        
    Returns:
        Configured TranslationPipeline instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    from pathlib import Path
    
    # Create pipeline configuration
    config = PipelineConfig(
        output_directory=Path(output_directory),
        target_language=target_language,
        tts_engine=tts_engine,
        **kwargs
    )
    
    # Initialize services
    video_scraping_service = YouTubeScrapingService(
        output_dir=config.output_directory / "scraped",
        audio_format=config.audio_format
    )
    
    # Translation service
    if openai_api_key:
        translation_service = OpenAITranslationService(openai_api_key, model=translation_model)
    else:
        translation_service = FallbackTranslationService()
    
    # TTS service
    if tts_engine == TTSEngine.OPENAI:
        if not openai_api_key:
            raise ConfigurationError("OpenAI API key required for OpenAI TTS")
        tts_service = create_tts_service(tts_engine, api_key=openai_api_key, model=tts_model)
    else:
        tts_service = create_tts_service(tts_engine)
    
    # Storage service
    storage_service = FileStorageService(config.output_directory)
    
    # Audio processing service
    audio_service = create_audio_processing_service()
    
    # Video processing service
    video_processing_service = FFmpegVideoProcessingService()
    
    # Transcript processing service
    transcript_service = HybridTranscriptProcessingService()
    
    return TranslationPipeline(
        video_scraping_service=video_scraping_service,
        transcript_service=transcript_service,
        translation_service=translation_service,
        tts_service=tts_service,
        audio_service=audio_service,
        video_processing_service=video_processing_service,
        storage_service=storage_service,
        config=config
    )


def create_simple_pipeline(openai_api_key: str = None) -> TranslationPipeline:
    """
    Create a simple pipeline with default settings.
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        Configured TranslationPipeline instance
    """
    return create_pipeline(
        openai_api_key=openai_api_key,
        target_language="es",
        tts_engine=TTSEngine.OPENAI if openai_api_key else TTSEngine.SYSTEM
    )


# Convenience exports
__all__ = [
    # Core models
    "VideoMetadata",
    "TranscriptSegment", 
    "ProcessedSegment",
    "TranslationSegment",
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingMode",
    "TTSEngine",
    "PipelineConfig",
    "TranslationJob",
    "AudioGenerationJob",
    
    # Main pipeline
    "TranslationPipeline",
    
    # Exceptions
    "PipelineError",
    "VideoScrapingError",
    "TranslationError",
    "TTSError",
    "ConfigurationError",
    "StorageError",
    "AudioProcessingError",
    "ValidationError",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "QuotaExceededError",
    
    # Configuration
    "PipelineSettings",
    "LanguageConfig",
    "TTSEngineConfig",
    "get_settings",
    "load_settings",
    "validate_configuration",
    
    # Services
    "YouTubeScrapingService",
    "HybridTranscriptProcessingService",
    "ProcessingConfig",
    "OpenAITranslationService",
    "FallbackTranslationService", 
    "OpenAITTSService",
    "SystemTTSService",
    "create_tts_service",
    "FileStorageService",
    "create_audio_processing_service",
    "FFmpegVideoProcessingService",
    
    # Utilities
    "configure_logging",
    "get_logger",
    "setup_pipeline_logging",
    
    # Factory functions
    "create_pipeline",
    "create_simple_pipeline",
]