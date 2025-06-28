"""Custom exceptions for the YouTube translation pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class VideoScrapingError(PipelineError):
    """Error during video scraping."""
    pass


class TranslationError(PipelineError):
    """Error during translation."""
    pass


class TTSError(PipelineError):
    """Error during text-to-speech generation."""
    pass


class ConfigurationError(PipelineError):
    """Error in configuration."""
    pass


class StorageError(PipelineError):
    """Error in storage operations."""
    pass


class AudioProcessingError(PipelineError):
    """Error during audio processing."""
    pass


class ValidationError(PipelineError):
    """Error in data validation."""
    pass


class APIError(PipelineError):
    """Error when calling external APIs."""
    
    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """Rate limit exceeded for external API."""
    pass


class AuthenticationError(APIError):
    """Authentication failed for external API."""
    pass


class QuotaExceededError(APIError):
    """API quota exceeded."""
    pass