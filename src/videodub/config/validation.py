"""Configuration validation utilities."""

import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from ..core.exceptions import ConfigurationError, ValidationError


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If URL is invalid
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValidationError(f"Invalid URL format: {url}")
        if result.scheme not in ("http", "https"):
            raise ValidationError(f"URL must use http or https: {url}")
        return True
    except Exception as e:
        raise ValidationError(f"URL validation failed: {e}")


def validate_video_url(url: str) -> bool:
    """
    Validate video URL format (YouTube, etc.).

    Args:
        url: Video URL to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If URL is invalid
    """
    validate_url(url)

    # Check for supported video platforms
    supported_domains = [
        "youtube.com",
        "youtu.be",
        "vimeo.com",
        "dailymotion.com",
        "twitch.tv",
        "tiktok.com",
        "instagram.com",
        "facebook.com",
        "twitter.com",
        "x.com",
        "reddit.com",
    ]

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    if not any(domain.endswith(supported) for supported in supported_domains):
        raise ValidationError(f"Unsupported video platform: {domain}")

    return True


def validate_language_code(language_code: str) -> bool:
    """
    Validate ISO 639-1 language code format.

    Args:
        language_code: Language code to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If language code is invalid
    """
    if not isinstance(language_code, str):
        raise ValidationError("Language code must be a string")

    if len(language_code) != 2:
        raise ValidationError("Language code must be exactly 2 characters")

    if not language_code.islower():
        raise ValidationError("Language code must be lowercase")

    if not language_code.isalpha():
        raise ValidationError("Language code must contain only letters")

    return True


def validate_file_path(
    file_path: str, must_exist: bool = False, must_be_writable: bool = False
) -> bool:
    """
    Validate file path.

    Args:
        file_path: File path to validate
        must_exist: Whether the file must already exist
        must_be_writable: Whether the path must be writable

    Returns:
        True if valid

    Raises:
        ValidationError: If path is invalid
    """
    if not isinstance(file_path, str):
        raise ValidationError("File path must be a string")

    if not file_path.strip():
        raise ValidationError("File path cannot be empty")

    path = Path(file_path)

    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if must_be_writable:
        if path.exists() and not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        # Check if parent directory is writable
        parent = path.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create parent directory: {e}")

        if not parent.is_dir():
            raise ValidationError(f"Parent is not a directory: {parent}")

        # Test write access
        try:
            test_file = parent / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValidationError(f"Directory is not writable: {e}")

    return True


def validate_directory_path(dir_path: str, create_if_missing: bool = False) -> bool:
    """
    Validate directory path.

    Args:
        dir_path: Directory path to validate
        create_if_missing: Whether to create directory if it doesn't exist

    Returns:
        True if valid

    Raises:
        ValidationError: If path is invalid
    """
    if not isinstance(dir_path, str):
        raise ValidationError("Directory path must be a string")

    if not dir_path.strip():
        raise ValidationError("Directory path cannot be empty")

    path = Path(dir_path)

    if path.exists() and not path.is_dir():
        raise ValidationError(f"Path exists but is not a directory: {dir_path}")

    if not path.exists() and create_if_missing:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create directory: {e}")

    if path.exists():
        # Test write access
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValidationError(f"Directory is not writable: {e}")

    return True


def validate_api_key(api_key: str, key_type: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate
        key_type: Type of API key (openai, google, azure)

    Returns:
        True if valid

    Raises:
        ValidationError: If API key is invalid
    """
    if not isinstance(api_key, str):
        raise ValidationError("API key must be a string")

    if not api_key.strip():
        raise ValidationError("API key cannot be empty")

    # Basic format validation based on key type
    if key_type.lower() == "openai":
        if not api_key.startswith("sk-"):
            raise ValidationError("OpenAI API key must start with 'sk-'")
        if len(api_key) < 20:
            raise ValidationError("OpenAI API key appears to be too short")

    elif key_type.lower() == "azure":
        if len(api_key) != 32:
            raise ValidationError("Azure API key must be exactly 32 characters")
        if not re.match("^[a-f0-9]+$", api_key):
            raise ValidationError(
                "Azure API key must contain only hexadecimal characters"
            )

    return True


def validate_text_length(text: str, max_length: int, field_name: str = "text") -> bool:
    """
    Validate text length.

    Args:
        text: Text to validate
        max_length: Maximum allowed length
        field_name: Name of the field being validated

    Returns:
        True if valid

    Raises:
        ValidationError: If text is too long
    """
    if not isinstance(text, str):
        raise ValidationError(f"{field_name} must be a string")

    if len(text) > max_length:
        raise ValidationError(
            f"{field_name} is too long: {len(text)} characters "
            f"(maximum: {max_length})"
        )

    return True


def validate_positive_number(value: int, field_name: str = "value") -> bool:
    """
    Validate positive number.

    Args:
        value: Number to validate
        field_name: Name of the field being validated

    Returns:
        True if valid

    Raises:
        ValidationError: If number is not positive
    """
    if not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")

    if value <= 0:
        raise ValidationError(f"{field_name} must be positive")

    return True


def validate_time_range(start_time: float, end_time: float) -> bool:
    """
    Validate time range.

    Args:
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        True if valid

    Raises:
        ValidationError: If time range is invalid
    """
    if not isinstance(start_time, (int, float)):
        raise ValidationError("Start time must be a number")

    if not isinstance(end_time, (int, float)):
        raise ValidationError("End time must be a number")

    if start_time < 0:
        raise ValidationError("Start time cannot be negative")

    if end_time <= start_time:
        raise ValidationError("End time must be greater than start time")

    return True


def validate_audio_format(
    audio_format: str, supported_formats: Optional[List[str]] = None
) -> bool:
    """
    Validate audio format.

    Args:
        audio_format: Audio format to validate
        supported_formats: List of supported formats

    Returns:
        True if valid

    Raises:
        ValidationError: If format is not supported
    """
    if not isinstance(audio_format, str):
        raise ValidationError("Audio format must be a string")

    audio_format = audio_format.lower().strip()

    if not audio_format:
        raise ValidationError("Audio format cannot be empty")

    # Default supported formats
    if supported_formats is None:
        supported_formats = ["wav", "mp3", "aac", "ogg", "flac", "opus", "webm"]

    if audio_format not in supported_formats:
        raise ValidationError(
            f"Unsupported audio format: {audio_format}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    return True
