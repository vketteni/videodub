"""Domain models for the YouTube translation pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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




class SourceType(Enum):
    """Source types for data extraction."""

    YOUTUBE = "youtube"
    TRANSCRIPTION = "transcription"
    LOCAL_FILE = "local_file"


class AlignmentStrategy(Enum):
    """Alignment strategies for timing synchronization."""

    LENGTH_BASED = "length_based"
    SENTENCE_BOUNDARY = "sentence_boundary"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"
    DYNAMIC_PROGRAMMING = "dynamic_programming"


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
class TimingMetadata:
    """Timing-related metadata for transcripts."""

    total_duration: float
    segment_count: int
    average_segment_duration: float
    timing_accuracy: Optional[float] = None  # 0.0-1.0 confidence score
    has_precise_timing: bool = True
    extraction_method: Optional[str] = None


@dataclass
class SpeechSegment:
    """Natural speech with timing - fundamental audio unit.
    
    Represents a segment of speech with precise millisecond timing.
    estimated_duration_ms is None until completed by AlignmentService.
    """

    start_time_ms: int
    estimated_duration_ms: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate speech segment data."""
        if self.start_time_ms < 0:
            raise ValueError("Start time cannot be negative")
        if self.estimated_duration_ms is not None and self.estimated_duration_ms <= 0:
            raise ValueError("Estimated duration must be positive")

    @property
    def end_time_ms(self) -> Optional[int]:
        """Get end time in milliseconds if duration is available."""
        if self.estimated_duration_ms is not None:
            return self.start_time_ms + self.estimated_duration_ms
        return None

    @property
    def has_complete_timing(self) -> bool:
        """Check if speech segment has complete timing information."""
        return self.estimated_duration_ms is not None

    @property
    def start_time_seconds(self) -> float:
        """Get start time in seconds for compatibility."""
        return self.start_time_ms / 1000.0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds if available."""
        if self.estimated_duration_ms is not None:
            return self.estimated_duration_ms / 1000.0
        return None


@dataclass
class TextContent:
    """Text content - original or translated.
    
    Pure text representation with no timing assumptions.
    """

    text: str
    language: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate text content."""
        if not self.text.strip():
            raise ValueError("Text cannot be empty")

    @property
    def word_count(self) -> int:
        """Get word count for the text."""
        return len(self.text.split())

    @property
    def character_count(self) -> int:
        """Get character count for the text."""
        return len(self.text)


@dataclass
class SpeechTextPair:
    """The relationship between speech and text.
    
    Represents the fundamental unit of the video dubbing pipeline:
    speech timing + text content + their relationship.
    """

    speech: SpeechSegment
    original_text: TextContent
    translated_text: Optional[TextContent] = None
    subtitle_display_duration_ms: Optional[int] = None  # Preserved YouTube metadata

    def __post_init__(self) -> None:
        """Validate speech-text pair."""
        if self.subtitle_display_duration_ms is not None and self.subtitle_display_duration_ms <= 0:
            raise ValueError("Subtitle display duration must be positive")

    @property
    def has_translation(self) -> bool:
        """Check if this pair has translated text."""
        return self.translated_text is not None

    @property
    def has_complete_timing(self) -> bool:
        """Check if speech timing is complete."""
        return self.speech.has_complete_timing

    @property
    def target_language(self) -> Optional[str]:
        """Get target language if translation exists."""
        if self.translated_text:
            return self.translated_text.language
        return None


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing information.
    
    DEPRECATED: Use SpeechTextPair instead.
    This model will be removed in Phase 2 of the integrated redesign.
    """

    start_time: float
    text: str
    end_time: Optional[float] = None
    subtitle_display_duration: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate segment data."""
        if self.start_time < 0:
            raise ValueError("Start time cannot be negative")
        if not self.text.strip():
            raise ValueError("Text cannot be empty")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if self.subtitle_display_duration is not None and self.subtitle_display_duration <= 0:
            raise ValueError("Subtitle display duration must be positive")

    @property
    def duration(self) -> Optional[float]:
        """Get segment duration in seconds if end_time is available."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def has_complete_timing(self) -> bool:
        """Check if segment has complete timing information."""
        return self.end_time is not None


@dataclass
class TimedTranscript:
    """Transcript with timing information from data extraction."""

    segments: List[TranscriptSegment]
    source_type: SourceType
    timing_metadata: TimingMetadata
    video_metadata: VideoMetadata
    language: Optional[str] = None
    extraction_quality: Optional[float] = None  # 0.0-1.0 quality score

    def __post_init__(self) -> None:
        """Validate timed transcript."""
        if not self.segments:
            raise ValueError("Timed transcript must have at least one segment")

        # Validate timing metadata consistency
        actual_duration = (
            max(s.end_time for s in self.segments) if self.segments else 0.0
        )
        if (
            abs(actual_duration - self.timing_metadata.total_duration) > 1.0
        ):  # 1 second tolerance
            raise ValueError(
                f"Timing metadata duration mismatch: {actual_duration} vs {self.timing_metadata.total_duration}"
            )

        if len(self.segments) != self.timing_metadata.segment_count:
            raise ValueError(
                f"Segment count mismatch: {len(self.segments)} vs {self.timing_metadata.segment_count}"
            )


@dataclass
class DataExtractionResult:
    """Result from data extraction containing transcript and file paths."""
    
    timed_transcript: TimedTranscript
    video_file_path: Optional[Path] = None
    audio_file_path: Optional[Path] = None
    subtitle_file_path: Optional[Path] = None


@dataclass
class TranslationSegment:
    """A translated segment with audio generation info.
    
    DEPRECATED: Use SpeechTextPair instead.
    This model will be removed in Phase 2 of the integrated redesign.
    """

    original_segment: "TranscriptSegment"  # Forward reference for transition
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
class AlignmentConfig:
    """Configuration for alignment strategy and parameters."""

    strategy: AlignmentStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment configuration."""
        # Set default parameters based on strategy
        if self.strategy == AlignmentStrategy.LENGTH_BASED:
            self.parameters.setdefault("length_weight", 0.8)
            self.parameters.setdefault("position_weight", 0.2)
        elif self.strategy == AlignmentStrategy.SENTENCE_BOUNDARY:
            self.parameters.setdefault("sentence_boundary_weight", 0.7)
            self.parameters.setdefault("length_weight", 0.3)
        elif self.strategy == AlignmentStrategy.SEMANTIC_SIMILARITY:
            self.parameters.setdefault("similarity_threshold", 0.6)
            self.parameters.setdefault(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
        elif self.strategy == AlignmentStrategy.HYBRID:
            self.parameters.setdefault("length_weight", 0.4)
            self.parameters.setdefault("boundary_weight", 0.3)
            self.parameters.setdefault("semantic_weight", 0.3)
        elif self.strategy == AlignmentStrategy.DYNAMIC_PROGRAMMING:
            self.parameters.setdefault("gap_penalty", 0.1)
            self.parameters.setdefault("mismatch_penalty", 0.2)


@dataclass
class AlignmentEvaluation:
    """Evaluation metrics for alignment quality assessment."""

    strategy: AlignmentStrategy
    timing_accuracy: float  # 0.0-1.0 how well timing is preserved
    text_preservation: float  # 0.0-1.0 how well text content is preserved
    boundary_alignment: float  # 0.0-1.0 sentence boundary alignment quality
    overall_score: float  # 0.0-1.0 weighted combination of metrics
    execution_time: float  # seconds taken for alignment
    segment_count: int
    average_confidence: float  # 0.0-1.0 average per-segment confidence

    def __post_init__(self) -> None:
        """Validate evaluation metrics."""
        metrics = [
            self.timing_accuracy,
            self.text_preservation,
            self.boundary_alignment,
            self.overall_score,
            self.average_confidence,
        ]
        for metric in metrics:
            if not 0.0 <= metric <= 1.0:
                raise ValueError(
                    f"Evaluation metric must be between 0.0 and 1.0, got {metric}"
                )

        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        if self.segment_count < 0:
            raise ValueError("Segment count cannot be negative")


@dataclass
class TimedTranslationSegment:
    """A translated segment with preserved timing information.
    
    DEPRECATED: Use SpeechTextPair instead.
    This model will be removed in Phase 2 of the integrated redesign.
    """

    start_time: float
    end_time: float
    original_text: str
    translated_text: str
    alignment_confidence: float  # 0.0-1.0 confidence score
    timing_adjustment: float = 0.0  # Timing adjustment applied (seconds)
    boundary_type: Optional[
        str
    ] = None  # "sentence_start", "sentence_end", "mid_sentence"

    def __post_init__(self) -> None:
        """Validate timed translation segment."""
        if self.start_time < 0:
            raise ValueError("Start time cannot be negative")
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if not self.original_text.strip():
            raise ValueError("Original text cannot be empty")
        if not self.translated_text.strip():
            raise ValueError("Translated text cannot be empty")
        if not 0.0 <= self.alignment_confidence <= 1.0:
            raise ValueError("Alignment confidence must be between 0.0 and 1.0")

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class TimedTranslation:
    """Translation with timing information preserved from original transcript."""

    segments: List[TimedTranslationSegment]
    original_transcript: TimedTranscript
    target_language: str
    alignment_config: AlignmentConfig
    alignment_evaluation: AlignmentEvaluation
    timing_metadata: TimingMetadata
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate timed translation."""
        if not self.segments:
            raise ValueError("Timed translation must have at least one segment")

        # Validate timing consistency
        if self.segments:
            total_duration = max(s.end_time for s in self.segments)
            original_duration = self.timing_metadata.total_duration
            if abs(total_duration - original_duration) > 2.0:  # 2 second tolerance
                raise ValueError(
                    f"Timing duration mismatch: {total_duration} vs {original_duration}"
                )

    @property
    def total_duration(self) -> float:
        """Get total duration of translation."""
        return max(s.end_time for s in self.segments) if self.segments else 0.0

    @property
    def average_confidence(self) -> float:
        """Get average alignment confidence across all segments."""
        if not self.segments:
            return 0.0
        return sum(s.alignment_confidence for s in self.segments) / len(self.segments)

    @property
    def alignment_strategy(self) -> AlignmentStrategy:
        """Get the alignment strategy used."""
        return self.alignment_config.strategy

    @property
    def overall_quality(self) -> float:
        """Get overall alignment quality score."""
        return self.alignment_evaluation.overall_score


