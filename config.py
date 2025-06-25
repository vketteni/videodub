"""
Configuration settings for the YouTube Translation Pipeline
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Pipeline settings
    DEFAULT_OUTPUT_DIR = "./pipeline_output"
    DEFAULT_TARGET_LANGUAGE = "es"  # Spanish
    DEFAULT_TTS_ENGINE = "openai"
    DEFAULT_AUDIO_FORMAT = "mp3"
    DEFAULT_VIDEO_QUALITY = "720p"
    
    # Supported languages
    SUPPORTED_LANGUAGES = {
        "es": "Spanish",
        "fr": "French", 
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi"
    }
    
    # TTS Engine settings
    TTS_ENGINES = {
        "openai": {
            "name": "OpenAI TTS",
            "api_key_env": "OPENAI_API_KEY",
            "models": ["tts-1", "tts-1-hd"],
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        },
        "google": {
            "name": "Google Cloud TTS",
            "api_key_env": "GOOGLE_APPLICATION_CREDENTIALS",
            "languages": list(SUPPORTED_LANGUAGES.keys())
        },
        "azure": {
            "name": "Azure Speech Services",
            "api_key_env": "AZURE_SPEECH_KEY",
            "region_env": "AZURE_SPEECH_REGION"
        },
        "system": {
            "name": "System TTS",
            "description": "Uses system espeak/say command"
        }
    }
    
    # API Keys (loaded from environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    # Translation API settings
    TRANSLATION_ENGINES = {
        "openai": {
            "name": "OpenAI GPT",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-3.5-turbo"
        },
        "google": {
            "name": "Google Translate",
            "api_key_env": "GOOGLE_APPLICATION_CREDENTIALS"
        },
        "azure": {
            "name": "Azure Translator",
            "api_key_env": "AZURE_TRANSLATOR_KEY",
            "region_env": "AZURE_TRANSLATOR_REGION"
        }
    }
    
    # Video scraper settings
    SCRAPER_CONFIG = {
        "audio_format": DEFAULT_AUDIO_FORMAT,
        "video_quality": DEFAULT_VIDEO_QUALITY,
        "extract_audio": True,
        "extract_transcript": True,
        "max_filesize": "500MB",
        "timeout": 300  # seconds
    }
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate_api_keys(cls, tts_engine: str = None, translation_engine: str = None):
        """Validate that required API keys are present"""
        missing_keys = []
        
        if tts_engine and tts_engine != "system":
            engine_config = cls.TTS_ENGINES.get(tts_engine, {})
            api_key_env = engine_config.get("api_key_env")
            if api_key_env and not os.getenv(api_key_env):
                missing_keys.append(api_key_env)
        
        if translation_engine:
            engine_config = cls.TRANSLATION_ENGINES.get(translation_engine, {})
            api_key_env = engine_config.get("api_key_env")
            if api_key_env and not os.getenv(api_key_env):
                missing_keys.append(api_key_env)
        
        return missing_keys
    
    @classmethod
    def get_language_name(cls, language_code: str) -> str:
        """Get full language name from code"""
        return cls.SUPPORTED_LANGUAGES.get(language_code, language_code.upper())
    
    @classmethod
    def is_language_supported(cls, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in cls.SUPPORTED_LANGUAGES