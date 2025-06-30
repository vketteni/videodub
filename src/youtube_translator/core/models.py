"""Domain models for the YouTube translation pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class ProcessingStatus(Enum):
    """Status of video processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TTSEngine(Enum):
    """Available text-to-speech engines."""
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    SYSTEM = "system"


@dataclass
class VideoMetadata:
    """Metadata for a video."""
    video_id: str
    title: str
    duration: float
    url: str
    channel: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    description: Optional[str] = None


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing information."""
    start_time: float
    end_time: float
    text: str

    def __post_init__(self) -> None:
        """Validate segment data."""
        if self.start_time < 0:
            raise ValueError("Start time cannot be negative")
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if not self.text.strip():
            raise ValueError("Text cannot be empty")


@dataclass
class TranslationSegment:
    """A translated segment with audio generation info."""
    original_segment: TranscriptSegment
    translated_text: str
    audio_path: Optional[Path] = None
    language: Optional[str] = None
    sentence_context: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Validate translation segment."""
        if not self.translated_text.strip():
            raise ValueError("Translated text cannot be empty")


@dataclass
class ProcessingResult:
    """Result of video processing."""
    video_id: str
    status: ProcessingStatus
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Optional[VideoMetadata] = None
    files: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    target_language: Optional[str] = None
    tts_engine: Optional[TTSEngine] = None
    cost_summary: Optional[Dict] = None

    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        if self.status != ProcessingStatus.FAILED:
            self.status = ProcessingStatus.FAILED

    def mark_completed(self) -> None:
        """Mark the processing as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.now()

    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark the processing as failed."""
        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.now()
        if error:
            self.add_error(error)


@dataclass
class PipelineConfig:
    """Configuration for the translation pipeline."""
    output_directory: Path
    target_language: str = "es"
    tts_engine: TTSEngine = TTSEngine.OPENAI
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    temp_directory: Optional[Path] = None
    audio_format: str = "wav"
    video_quality: str = "720p"
    extract_audio: bool = True
    extract_transcript: bool = True

    def __post_init__(self) -> None:
        """Validate and set up configuration."""
        # Ensure directories are Path objects
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        if self.temp_directory and isinstance(self.temp_directory, str):
            self.temp_directory = Path(self.temp_directory)

        # Create directories if they don't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if self.temp_directory:
            self.temp_directory.mkdir(parents=True, exist_ok=True)


@dataclass
class TranslationJob:
    """A translation job containing all segments to process."""
    video_id: str
    segments: List[TranscriptSegment]
    target_language: str
    tts_engine: TTSEngine
    output_directory: Path
    created_at: datetime = field(default_factory=datetime.now)
    translated_segments: List[TranslationSegment] = field(default_factory=list)

    def add_translated_segment(self, segment: TranslationSegment) -> None:
        """Add a translated segment to the job."""
        self.translated_segments.append(segment)

    @property
    def total_segments(self) -> int:
        """Total number of segments to translate."""
        return len(self.segments)

    @property
    def completed_segments(self) -> int:
        """Number of completed translations."""
        return len(self.translated_segments)

    @property
    def progress_percentage(self) -> float:
        """Progress as a percentage."""
        if self.total_segments == 0:
            return 0.0
        return (self.completed_segments / self.total_segments) * 100.0


@dataclass
class AudioGenerationJob:
    """Job for generating audio from translated segments."""
    segments: List[TranslationSegment]
    output_directory: Path
    language: str
    tts_engine: TTSEngine
    audio_format: str = "wav"
    created_at: datetime = field(default_factory=datetime.now)
    generated_files: List[Path] = field(default_factory=list)

    def add_generated_file(self, file_path: Path) -> None:
        """Add a generated audio file."""
        self.generated_files.append(file_path)

    @property
    def is_complete(self) -> bool:
        """Check if all segments have been processed."""
        return len(self.generated_files) >= len(self.segments)