"""Abstract interfaces for the YouTube translation pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, AsyncIterator, Dict

from .models import (
    VideoMetadata,
    TranscriptSegment,
    ProcessedSegment,
    TranslationSegment,
    ProcessingResult,
    TTSEngine,
    TranslationJob,
    AudioGenerationJob,
    ProcessingMode,
)


class VideoScrapingService(ABC):
    """Abstract interface for video scraping services."""

    @abstractmethod
    async def scrape_video(self, url: str) -> Tuple[VideoMetadata, List[TranscriptSegment]]:
        """
        Scrape video and extract metadata and transcript.
        
        Args:
            url: Video URL to scrape
            
        Returns:
            Tuple of video metadata and transcript segments
            
        Raises:
            VideoScrapingError: If scraping fails
        """
        pass

    @abstractmethod
    async def scrape_audio_only(self, url: str) -> Tuple[VideoMetadata, Path]:
        """
        Scrape only audio from video.
        
        Args:
            url: Video URL to scrape
            
        Returns:
            Tuple of video metadata and audio file path
            
        Raises:
            VideoScrapingError: If scraping fails
        """
        pass

    @abstractmethod
    async def get_transcript(self, url: str) -> List[TranscriptSegment]:
        """
        Extract transcript from video.
        
        Args:
            url: Video URL
            
        Returns:
            List of transcript segments
            
        Raises:
            VideoScrapingError: If transcript extraction fails
        """
        pass


class TranslationService(ABC):
    """Abstract interface for translation services."""

    @abstractmethod
    async def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate a single text string.
        
        Args:
            text: Text to translate
            target_language: Target language code
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        pass

    @abstractmethod
    async def translate_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_language: str
    ) -> List[TranslationSegment]:
        """
        Translate multiple transcript segments.
        
        Args:
            segments: List of transcript segments to translate
            target_language: Target language code
            
        Returns:
            List of translated segments
            
        Raises:
            TranslationError: If translation fails
        """
        pass

    @abstractmethod
    async def translate_batch(
        self, 
        job: TranslationJob
    ) -> AsyncIterator[TranslationSegment]:
        """
        Translate segments in batches with progress tracking.
        
        Args:
            job: Translation job with segments to process
            
        Yields:
            Translated segments as they are completed
            
        Raises:
            TranslationError: If translation fails
        """
        pass


class TTSService(ABC):
    """Abstract interface for text-to-speech services."""

    @abstractmethod
    async def generate_audio(
        self, 
        text: str, 
        language: str,
        output_path: Path,
        voice: Optional[str] = None
    ) -> Path:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            language: Language code for TTS
            output_path: Where to save the audio file
            voice: Optional voice identifier
            
        Returns:
            Path to generated audio file
            
        Raises:
            TTSError: If audio generation fails
        """
        pass

    @abstractmethod
    async def generate_batch_audio(
        self, 
        job: AudioGenerationJob
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
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages for this TTS engine.
        
        Returns:
            List of language codes
        """
        pass

    @abstractmethod
    def get_supported_voices(self, language: str) -> List[str]:
        """
        Get list of available voices for a language.
        
        Args:
            language: Language code
            
        Returns:
            List of voice identifiers
        """
        pass


class StorageService(ABC):
    """Abstract interface for storage services."""

    @abstractmethod
    async def save_metadata(self, video_id: str, metadata: VideoMetadata) -> Path:
        """
        Save video metadata to storage.
        
        Args:
            video_id: Video identifier
            metadata: Video metadata to save
            
        Returns:
            Path where metadata was saved
        """
        pass

    @abstractmethod
    async def save_translation_data(
        self, 
        video_id: str, 
        segments: List[TranslationSegment]
    ) -> Path:
        """
        Save translation data to storage.
        
        Args:
            video_id: Video identifier
            segments: Translated segments to save
            
        Returns:
            Path where translation data was saved
        """
        pass

    @abstractmethod
    async def save_processing_result(self, result: ProcessingResult) -> Path:
        """
        Save processing result to storage.
        
        Args:
            result: Processing result to save
            
        Returns:
            Path where result was saved
        """
        pass

    @abstractmethod
    async def load_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """
        Load video metadata from storage.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video metadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def load_translation_data(self, video_id: str) -> Optional[List[TranslationSegment]]:
        """
        Load translation data from storage.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Translation segments if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_video_directory(self, video_id: str) -> Path:
        """
        Get the storage directory for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Path to video directory
        """
        pass


class TranscriptProcessingService(ABC):
    """Abstract interface for transcript processing services."""

    @abstractmethod
    async def process_transcript(
        self, 
        segments: List[TranscriptSegment],
        mode: ProcessingMode = ProcessingMode.HYBRID
    ) -> List[ProcessedSegment]:
        """
        Process and reconstruct transcript segments for optimal translation.
        
        Args:
            segments: Raw transcript segments to process
            mode: Processing mode (rule-based, AI-enhanced, or hybrid)
            
        Returns:
            Processed transcript segments ready for translation
            
        Raises:
            TranscriptProcessingError: If processing fails
        """
        pass

    @abstractmethod
    async def validate_segments(self, segments: List[TranscriptSegment]) -> bool:
        """
        Validate transcript segments for quality and completeness.
        
        Args:
            segments: Transcript segments to validate
            
        Returns:
            True if segments are valid
        """
        pass


class AudioProcessingService(ABC):
    """Abstract interface for audio processing services."""

    @abstractmethod
    async def combine_audio_segments(
        self, 
        audio_files: List[Path], 
        output_path: Path,
        segments: List[TranslationSegment]
    ) -> Path:
        """
        Combine multiple audio segments into a single file.
        
        Args:
            audio_files: List of audio file paths
            output_path: Where to save combined audio
            segments: Translation segments with timing info
            
        Returns:
            Path to combined audio file
        """
        pass

    @abstractmethod
    async def adjust_audio_timing(
        self, 
        audio_path: Path, 
        segment: TranslationSegment
    ) -> Path:
        """
        Adjust audio timing to match original segment duration.
        
        Args:
            audio_path: Path to audio file
            segment: Translation segment with timing info
            
        Returns:
            Path to adjusted audio file
        """
        pass


class VideoProcessingService(ABC):
    """Abstract interface for video processing and dubbing services."""

    @abstractmethod
    async def create_dubbed_video(
        self, 
        original_video_path: Path, 
        translated_audio_path: Path, 
        output_path: Path,
        segments: List[TranslationSegment]
    ) -> Path:
        """
        Create dubbed video by combining original video with translated audio.
        
        Args:
            original_video_path: Path to original video file
            translated_audio_path: Path to translated audio file
            output_path: Where to save dubbed video
            segments: Translation segments with timing info
            
        Returns:
            Path to dubbed video file
        """
        pass

    @abstractmethod
    async def extract_video_metadata(self, video_path: Path) -> Dict:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata (duration, resolution, etc.)
        """
        pass


class ProgressTracker(ABC):
    """Abstract interface for progress tracking."""

    @abstractmethod
    async def start_processing(self, video_id: str, total_steps: int) -> None:
        """Start tracking progress for a video."""
        pass

    @abstractmethod
    async def update_progress(self, video_id: str, completed_steps: int) -> None:
        """Update progress for a video."""
        pass

    @abstractmethod
    async def complete_processing(self, video_id: str) -> None:
        """Mark processing as complete."""
        pass

    @abstractmethod
    async def fail_processing(self, video_id: str, error: str) -> None:
        """Mark processing as failed."""
        pass