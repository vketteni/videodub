"""Text-to-speech service implementations."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import AsyncIterator, List, Optional

import structlog

from ..adapters.openai_adapter import OpenAITTSAdapter
from ..config.validation import validate_audio_format, validate_text_length
from ..core.exceptions import APIError, TTSError
from ..core.interfaces import TTSService
from ..core.models import AudioGenerationJob, TTSEngine

logger = structlog.get_logger(__name__)


class OpenAITTSService(TTSService):
    """Text-to-speech service using OpenAI API."""

    def __init__(
        self, api_key: str, default_voice: str = "alloy", model: str = "tts-1"
    ):
        """
        Initialize OpenAI TTS service.

        Args:
            api_key: OpenAI API key
            default_voice: Default voice to use
            model: TTS model to use
        """
        self.adapter = OpenAITTSAdapter(api_key, default_voice, model)
        self.max_text_length = 4096
        self.supported_languages = [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ja",
            "ko",
            "zh",
            "ru",
            "ar",
            "hi",
        ]
        self.supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

        logger.info(
            "OpenAI TTS service initialized", default_voice=default_voice, model=model
        )

    async def generate_audio(
        self, text: str, language: str, output_path: Path, voice: Optional[str] = None
    ) -> Path:
        """
        Generate audio from text.

        Args:
            text: Text to convert to speech
            language: Language code (not used by OpenAI TTS, but kept for interface compatibility)
            output_path: Where to save the audio file
            voice: Voice to use (optional)

        Returns:
            Path to generated audio file

        Raises:
            TTSError: If audio generation fails
        """
        validate_text_length(text, self.max_text_length, "TTS input text")

        if not text.strip():
            raise TTSError("Text cannot be empty")

        logger.debug(
            "Generating audio",
            text_length=len(text),
            language=language,
            voice=voice,
            output_path=str(output_path),
        )

        try:
            # Determine audio format from file extension
            audio_format = output_path.suffix.lstrip(".").lower()
            if not audio_format:
                audio_format = "wav"

            validate_audio_format(audio_format, self.supported_formats)

            # Generate speech
            result_path = await self.adapter.generate_speech_file(
                text=text,
                output_path=output_path,
                voice=voice,
                response_format=audio_format,
            )

            logger.debug("Audio generation completed", output_path=str(result_path))
            return result_path

        except APIError as e:
            logger.error("OpenAI TTS API error", error=str(e))
            raise TTSError(
                f"TTS generation failed: {e.message}", details={"api_error": str(e)}
            )
        except Exception as e:
            logger.error("Audio generation failed", error=str(e))
            raise TTSError(f"Unexpected error during TTS: {str(e)}")

    async def generate_batch_audio(
        self, job: AudioGenerationJob
    ) -> AsyncIterator[Path]:
        """
        Generate audio for multiple segments.

        Args:
            job: Audio generation job

        Yields:
            Paths to generated audio files

        Raises:
            TTSError: If audio generation fails
        """
        if not job.segments:
            return

        logger.info(
            "Starting batch audio generation",
            segment_count=len(job.segments),
            language=job.language,
            tts_engine=job.tts_engine.value,
        )

        job.output_directory.mkdir(parents=True, exist_ok=True)

        # Process segments with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def generate_segment_audio(index: int, segment) -> Path:
            async with semaphore:
                try:
                    filename = f"segment_{index:04d}.{job.audio_format}"
                    output_path = job.output_directory / filename

                    await self.generate_audio(
                        text=segment.translated_text,
                        language=job.language,
                        output_path=output_path,
                    )

                    job.add_generated_file(output_path)
                    return output_path

                except Exception as e:
                    logger.error(
                        "Failed to generate audio for segment",
                        segment_index=index,
                        error=str(e),
                    )
                    # Create empty file as placeholder
                    output_path = (
                        job.output_directory
                        / f"segment_{index:04d}_failed.{job.audio_format}"
                    )
                    output_path.touch()
                    return output_path

        # Execute generation tasks
        tasks = [
            generate_segment_audio(i, segment) for i, segment in enumerate(job.segments)
        ]

        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            audio_path = await task
            yield audio_path

        logger.info(
            "Batch audio generation completed",
            total_segments=len(job.segments),
            generated_files=len(job.generated_files),
            success_rate=len(job.generated_files) / len(job.segments) * 100,
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()

    def get_supported_voices(self, language: str) -> List[str]:
        """Get list of available voices (OpenAI voices are language-agnostic)."""
        return self.adapter.available_voices.copy()


class SystemTTSService(TTSService):
    """Fallback TTS service using system commands (espeak, say)."""

    def __init__(self):
        """Initialize system TTS service."""
        self.max_text_length = 500  # Conservative limit for system TTS
        self.supported_formats = ["wav"]
        self._tts_command = self._detect_tts_command()

        logger.info("System TTS service initialized", command=self._tts_command)

    def _detect_tts_command(self) -> Optional[str]:
        """Detect available system TTS command."""
        # Check for espeak (Linux)
        if subprocess.run(["which", "espeak"], capture_output=True).returncode == 0:
            return "espeak"

        # Check for say (macOS)
        if subprocess.run(["which", "say"], capture_output=True).returncode == 0:
            return "say"

        return None

    async def generate_audio(
        self, text: str, language: str, output_path: Path, voice: Optional[str] = None
    ) -> Path:
        """
        Generate audio using system TTS.

        Args:
            text: Text to convert to speech
            language: Language code
            output_path: Where to save the audio file
            voice: Voice to use (ignored for system TTS)

        Returns:
            Path to generated audio file

        Raises:
            TTSError: If audio generation fails
        """
        validate_text_length(text, self.max_text_length, "System TTS input text")

        if not text.strip():
            raise TTSError("Text cannot be empty")

        if not self._tts_command:
            # Create placeholder file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(f"# System TTS placeholder for: {text[:50]}...")
            return output_path

        logger.debug(
            "Generating audio with system TTS",
            command=self._tts_command,
            text_length=len(text),
            output_path=str(output_path),
        )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Execute system TTS command
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_tts_command, text, output_path)

            logger.debug("System TTS completed", output_path=str(output_path))
            return output_path

        except Exception as e:
            logger.error("System TTS failed", error=str(e))
            raise TTSError(f"System TTS generation failed: {str(e)}")

    def _run_tts_command(self, text: str, output_path: Path) -> None:
        """Run the system TTS command synchronously."""
        if self._tts_command == "espeak":
            cmd = ["espeak", "-s", "150", "-w", str(output_path), text]
        elif self._tts_command == "say":
            # macOS say command outputs to AIFF by default
            aiff_path = output_path.with_suffix(".aiff")
            cmd = ["say", text, "-o", str(aiff_path)]
        else:
            raise TTSError("No system TTS command available")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise TTSError(f"TTS command failed: {result.stderr}")

        # Convert AIFF to WAV if needed (macOS)
        if self._tts_command == "say" and output_path.suffix.lower() == ".wav":
            try:
                import pydub

                audio = pydub.AudioSegment.from_file(str(aiff_path))
                audio.export(str(output_path), format="wav")
                aiff_path.unlink()  # Remove temporary AIFF file
            except ImportError:
                # If pydub not available, just rename the file
                aiff_path.rename(output_path)

    async def generate_batch_audio(
        self, job: AudioGenerationJob
    ) -> AsyncIterator[Path]:
        """
        Generate audio for multiple segments using system TTS.

        Args:
            job: Audio generation job

        Yields:
            Paths to generated audio files
        """
        logger.info(
            "Starting system TTS batch generation",
            segment_count=len(job.segments),
            command=self._tts_command,
        )

        for i, segment in enumerate(job.segments):
            try:
                filename = f"segment_{i:04d}.{job.audio_format}"
                output_path = job.output_directory / filename

                await self.generate_audio(
                    text=segment.translated_text,
                    language=job.language,
                    output_path=output_path,
                )

                job.add_generated_file(output_path)
                yield output_path

            except Exception as e:
                logger.error(
                    "Failed to generate audio for segment",
                    segment_index=i,
                    error=str(e),
                )
                # Create placeholder file
                output_path = (
                    job.output_directory / f"segment_{i:04d}_failed.{job.audio_format}"
                )
                output_path.touch()
                yield output_path

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages (limited for system TTS)."""
        return ["en"]  # System TTS typically only supports English well

    def get_supported_voices(self, language: str) -> List[str]:
        """Get list of available voices (system default only)."""
        return ["default"]


def create_tts_service(engine: TTSEngine, **kwargs) -> TTSService:
    """
    Factory function to create appropriate TTS service.

    Args:
        engine: TTS engine to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured TTS service

    Raises:
        TTSError: If service cannot be created
    """
    if engine == TTSEngine.OPENAI:
        api_key = kwargs.get("api_key")
        if not api_key:
            raise TTSError("OpenAI API key required for OpenAI TTS")

        return OpenAITTSService(
            api_key=api_key,
            default_voice=kwargs.get("default_voice", "alloy"),
            model=kwargs.get("model", "tts-1"),
        )

    elif engine == TTSEngine.SYSTEM:
        return SystemTTSService()

    elif engine == TTSEngine.GOOGLE:
        # TODO: Implement Google TTS service
        raise TTSError("Google TTS service not yet implemented")

    elif engine == TTSEngine.AZURE:
        # TODO: Implement Azure TTS service
        raise TTSError("Azure TTS service not yet implemented")

    else:
        raise TTSError(f"Unsupported TTS engine: {engine}")
