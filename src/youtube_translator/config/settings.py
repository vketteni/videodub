"""Configuration management for the YouTube translation pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict

from ..core.models import TTSEngine
from ..core.exceptions import ConfigurationError


class PipelineSettings(BaseSettings):
    """Main configuration settings for the pipeline."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    google_credentials_path: Optional[str] = Field(None, description="Google Cloud credentials file path")
    azure_speech_key: Optional[str] = Field(None, description="Azure Speech Services key")
    azure_speech_region: str = Field("eastus", description="Azure Speech Services region")
    azure_translator_key: Optional[str] = Field(None, description="Azure Translator key")
    azure_translator_region: Optional[str] = Field(None, description="Azure Translator region")

    # Processing Configuration
    default_target_language: str = Field("es", description="Default target language code")
    default_tts_engine: TTSEngine = Field(TTSEngine.OPENAI, description="Default TTS engine")
    max_concurrent_requests: int = Field(5, ge=1, le=20, description="Maximum concurrent API requests")
    request_timeout: int = Field(30, ge=5, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(3, ge=0, le=10, description="Number of retry attempts for failed requests")
    
    # Storage Configuration
    output_directory: str = Field("./output", description="Output directory for processed files")
    temp_directory: Optional[str] = Field(None, description="Temporary directory for processing")
    keep_temp_files: bool = Field(False, description="Whether to keep temporary files after processing")
    
    # Video Processing Configuration
    audio_format: str = Field("wav", description="Audio format for generated files")
    video_quality: str = Field("720p", description="Video quality for downloads")
    extract_audio: bool = Field(True, description="Whether to extract audio from videos")
    extract_transcript: bool = Field(True, description="Whether to extract transcripts")
    max_video_duration: int = Field(3600, ge=60, description="Maximum video duration in seconds")
    
    # Logging Configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format"
    )
    enable_structured_logging: bool = Field(True, description="Enable structured JSON logging")
    
    # Feature Flags
    enable_caching: bool = Field(True, description="Enable response caching")
    enable_metrics: bool = Field(False, description="Enable metrics collection")
    enable_health_checks: bool = Field(True, description="Enable health check endpoints")

    @validator('openai_api_key')
    def validate_openai_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate OpenAI API key format."""
        if v is not None and not v.startswith('sk-'):
            raise ValueError('OpenAI API key must start with "sk-"')
        return v

    @validator('default_target_language')
    def validate_language_code(cls, v: str) -> str:
        """Validate language code format."""
        if len(v) != 2 or not v.islower():
            raise ValueError('Language code must be a 2-letter lowercase code (e.g., "es", "fr")')
        return v

    @validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {", ".join(valid_levels)}')
        return v.upper()

    def get_output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_directory).resolve()

    def get_temp_path(self) -> Path:
        """Get temporary directory as Path object."""
        if self.temp_directory:
            return Path(self.temp_directory).resolve()
        return self.get_output_path() / "temp"

    def validate_api_keys(self, required_engines: Optional[Set[TTSEngine]] = None) -> List[str]:
        """
        Validate that required API keys are present.
        
        Args:
            required_engines: Set of TTS engines that require validation
            
        Returns:
            List of missing API key names
        """
        missing_keys = []
        
        if not required_engines:
            required_engines = {self.default_tts_engine}
        
        for engine in required_engines:
            if engine == TTSEngine.OPENAI and not self.openai_api_key:
                missing_keys.append("OPENAI_API_KEY")
            elif engine == TTSEngine.GOOGLE and not self.google_credentials_path:
                missing_keys.append("GOOGLE_APPLICATION_CREDENTIALS")
            elif engine == TTSEngine.AZURE and not self.azure_speech_key:
                missing_keys.append("AZURE_SPEECH_KEY")
        
        return missing_keys

    def create_directories(self) -> None:
        """Create necessary directories."""
        self.get_output_path().mkdir(parents=True, exist_ok=True)
        self.get_temp_path().mkdir(parents=True, exist_ok=True)


class LanguageConfig:
    """Configuration for supported languages and their settings."""
    
    SUPPORTED_LANGUAGES = {
        "es": {"name": "Spanish", "tts_voice": "alloy"},
        "fr": {"name": "French", "tts_voice": "echo"},
        "de": {"name": "German", "tts_voice": "fable"},
        "it": {"name": "Italian", "tts_voice": "onyx"},
        "pt": {"name": "Portuguese", "tts_voice": "nova"},
        "ja": {"name": "Japanese", "tts_voice": "shimmer"},
        "ko": {"name": "Korean", "tts_voice": "alloy"},
        "zh": {"name": "Chinese", "tts_voice": "echo"},
        "ru": {"name": "Russian", "tts_voice": "fable"},
        "ar": {"name": "Arabic", "tts_voice": "onyx"},
        "hi": {"name": "Hindi", "tts_voice": "nova"},
    }
    
    @classmethod
    def get_language_name(cls, language_code: str) -> str:
        """Get full language name from code."""
        return cls.SUPPORTED_LANGUAGES.get(language_code, {}).get("name", language_code.upper())
    
    @classmethod
    def get_default_voice(cls, language_code: str) -> str:
        """Get default TTS voice for language."""
        return cls.SUPPORTED_LANGUAGES.get(language_code, {}).get("tts_voice", "alloy")
    
    @classmethod
    def is_language_supported(cls, language_code: str) -> bool:
        """Check if language is supported."""
        return language_code in cls.SUPPORTED_LANGUAGES
    
    @classmethod
    def get_supported_language_codes(cls) -> List[str]:
        """Get list of supported language codes."""
        return list(cls.SUPPORTED_LANGUAGES.keys())


class TTSEngineConfig:
    """Configuration for TTS engines."""
    
    ENGINE_CONFIGS = {
        TTSEngine.OPENAI: {
            "name": "OpenAI TTS",
            "api_key_env": "OPENAI_API_KEY",
            "models": ["tts-1", "tts-1-hd"],
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "max_text_length": 4096,
            "supported_formats": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
        },
        TTSEngine.GOOGLE: {
            "name": "Google Cloud TTS",
            "api_key_env": "GOOGLE_APPLICATION_CREDENTIALS",
            "max_text_length": 5000,
            "supported_formats": ["wav", "mp3", "ogg"],
        },
        TTSEngine.AZURE: {
            "name": "Azure Speech Services",
            "api_key_env": "AZURE_SPEECH_KEY",
            "region_env": "AZURE_SPEECH_REGION",
            "max_text_length": 1000,
            "supported_formats": ["wav", "mp3", "ogg", "webm"],
        },
        TTSEngine.SYSTEM: {
            "name": "System TTS",
            "description": "Uses system espeak/say command",
            "max_text_length": 500,
            "supported_formats": ["wav"],
        }
    }
    
    @classmethod
    def get_engine_config(cls, engine: TTSEngine) -> Dict:
        """Get configuration for a TTS engine."""
        return cls.ENGINE_CONFIGS.get(engine, {})
    
    @classmethod
    def get_max_text_length(cls, engine: TTSEngine) -> int:
        """Get maximum text length for an engine."""
        return cls.get_engine_config(engine).get("max_text_length", 1000)
    
    @classmethod
    def get_supported_formats(cls, engine: TTSEngine) -> List[str]:
        """Get supported audio formats for an engine."""
        return cls.get_engine_config(engine).get("supported_formats", ["wav"])


def load_settings() -> PipelineSettings:
    """Load and validate settings."""
    try:
        settings = PipelineSettings()
        settings.create_directories()
        return settings
    except Exception as e:
        raise ConfigurationError(f"Failed to load settings: {str(e)}")


def validate_configuration(settings: PipelineSettings) -> None:
    """
    Validate complete configuration.
    
    Args:
        settings: Settings to validate
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Validate language support
    if not LanguageConfig.is_language_supported(settings.default_target_language):
        raise ConfigurationError(
            f"Unsupported language: {settings.default_target_language}. "
            f"Supported languages: {', '.join(LanguageConfig.get_supported_language_codes())}"
        )
    
    # Validate TTS engine configuration
    missing_keys = settings.validate_api_keys({settings.default_tts_engine})
    if missing_keys:
        raise ConfigurationError(
            f"Missing required API keys for {settings.default_tts_engine.value}: {', '.join(missing_keys)}"
        )
    
    # Validate directories are writable
    try:
        output_path = settings.get_output_path()
        temp_file = output_path / "test_write.tmp"
        temp_file.write_text("test")
        temp_file.unlink()
    except Exception as e:
        raise ConfigurationError(f"Output directory is not writable: {e}")


# Global settings instance
_settings: Optional[PipelineSettings] = None


def get_settings() -> PipelineSettings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings