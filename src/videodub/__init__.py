"""YouTube Translation Pipeline - A comprehensive AI-powered video translation system."""

__version__ = "1.0.0"
__author__ = "Vincent"
__email__ = "vincent@example.com"

# Configuration
from .config.settings import (
    LanguageConfig,
    PipelineSettings,
    TTSEngineConfig,
    get_settings,
    load_settings,
    validate_configuration,
)
from .core.exceptions import (
    APIError,
    AudioProcessingError,
    AuthenticationError,
    ConfigurationError,
    DataExtractionError,
    AlignmentError,
    PipelineError,
    QuotaExceededError,
    RateLimitError,
    StorageError,
    TranslationError,
    TTSError,
    ValidationError,
)

# Core imports
from .core.models import (
    DataExtractionResult,
    PipelineConfig,
    ProcessingResult,
    ProcessingStatus,
    TranscriptSegment,
    TranslationSegment,
    TTSEngine,
    VideoMetadata,
)
# Services
from .services.audio import create_audio_processing_service
from .services.storage import FileStorageService
from .services.translator import FallbackTranslationService, OpenAITranslationService
from .services.tts import OpenAITTSService, SystemTTSService, create_tts_service
from .services.video import FFmpegVideoProcessingService

# Utilities
from .utils.logging import configure_logging, get_logger, setup_pipeline_logging


# Legacy factory functions - deprecated
# Use NewTranslationPipeline from src.videodub.core.new_pipeline instead
# def create_pipeline(...):
#     """Legacy pipeline factory - use NewTranslationPipeline instead"""
#     pass


# Convenience exports
__all__ = [
    # Core models
    "DataExtractionResult",
    "VideoMetadata",
    "TranscriptSegment",
    "TranslationSegment",
    "ProcessingResult",
    "ProcessingStatus",
    "TTSEngine",
    "PipelineConfig",
    # Exceptions
    "PipelineError",
    "DataExtractionError",
    "AlignmentError",
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
    # Services (active)
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
    # Main pipeline available at: from videodub.core.pipeline import TranslationPipeline
]
