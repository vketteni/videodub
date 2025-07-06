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


class ProcessingMode(Enum):
    """Processing modes for transcript processing."""

    RULE_BASED = "rule_based"
    AI_ENHANCED = "ai_enhanced"
    HYBRID = "hybrid"


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
class ProcessedSegment:
    """A processed transcript segment with enhancement metadata."""

    merged_segments: List[TranscriptSegment]
    processed_text: str
    processing_mode: ProcessingMode
    sequence_number: int
    original_indices: List[int] = field(default_factory=list)
    is_sentence_complete: bool = False
    context_quality_score: Optional[float] = None
    timing_preserved: bool = True
    enhancement_metadata: Dict[str, Any] = field(default_factory=dict)
    ready_for_translation: bool = True

    def __post_init__(self) -> None:
        """Validate processed segment."""
        if not self.processed_text.strip():
            raise ValueError("Processed text cannot be empty")
        if not self.merged_segments:
            raise ValueError("Must have at least one merged segment")
        if self.sequence_number < 0:
            raise ValueError("Sequence number must be non-negative")
        if self.context_quality_score is not None:
            if not 0.0 <= self.context_quality_score <= 1.0:
                raise ValueError("Context quality score must be between 0.0 and 1.0")

        # Auto-populate original_indices if not provided
        if not self.original_indices:
            # This is a fallback - in practice, original_indices should be set explicitly
            self.original_indices = list(range(len(self.merged_segments)))

    @property
    def start_time(self) -> float:
        """Get the earliest start time from merged segments."""
        return min(s.start_time for s in self.merged_segments)

    @property
    def end_time(self) -> float:
        """Get the latest end time from merged segments."""
        return max(s.end_time for s in self.merged_segments)

    @property
    def duration(self) -> float:
        """Get the total duration of the processed segment."""
        return self.end_time - self.start_time


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
    """A translated segment with preserved timing information."""

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
