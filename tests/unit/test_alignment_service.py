"""Tests for alignment services."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from videodub.core.exceptions import AlignmentError
from videodub.core.models import (
    AlignmentConfig,
    AlignmentEvaluation,
    AlignmentStrategy,
    SourceType,
    TimedTranscript,
    TimedTranslation,
    TimedTranslationSegment,
    TimingMetadata,
    TranscriptSegment,
    VideoMetadata,
)
from videodub.services.alignment import TimingAlignmentService


class TestTimingAlignmentService:
    """Test TimingAlignmentService implementation."""

    @pytest.fixture
    def alignment_service(self):
        """Create TimingAlignmentService instance."""
        return TimingAlignmentService()

    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            video_id="test_video",
            title="Test Video",
            duration=120.0,
            url="https://example.com/video",
            channel="Test Channel",
        )

    @pytest.fixture
    def sample_timing_metadata(self):
        """Create sample timing metadata."""
        return TimingMetadata(
            total_duration=120.0,
            segment_count=3,
            average_segment_duration=40.0,
            timing_accuracy=0.95,
            has_precise_timing=True,
            extraction_method="youtube_api",
        )

    @pytest.fixture
    def sample_transcript_segments(self):
        """Create sample transcript segments."""
        return [
            TranscriptSegment(start_time=0.0, end_time=40.0, text="Hello world"),
            TranscriptSegment(start_time=40.0, end_time=80.0, text="This is a test"),
            TranscriptSegment(start_time=80.0, end_time=120.0, text="Goodbye everyone"),
        ]

    @pytest.fixture
    def sample_timed_transcript(
        self, sample_transcript_segments, sample_timing_metadata, sample_video_metadata
    ):
        """Create sample timed transcript."""
        return TimedTranscript(
            segments=sample_transcript_segments,
            source_type=SourceType.YOUTUBE,
            timing_metadata=sample_timing_metadata,
            video_metadata=sample_video_metadata,
            language="en",
            extraction_quality=0.9,
        )

    @pytest.fixture
    def sample_translated_texts(self):
        """Create sample translated texts."""
        return [
            "Hola mundo",
            "Esto es una prueba",
            "Adiós a todos.",  # Add period to trigger sentence boundary detection
        ]

    @pytest.fixture
    def length_based_config(self):
        """Create length-based alignment config."""
        return AlignmentConfig(
            strategy=AlignmentStrategy.LENGTH_BASED,
            parameters={"length_weight": 0.8, "position_weight": 0.2},
        )

    @pytest.fixture
    def sentence_boundary_config(self):
        """Create sentence boundary alignment config."""
        return AlignmentConfig(
            strategy=AlignmentStrategy.SENTENCE_BOUNDARY,
            parameters={"sentence_boundary_weight": 0.7, "length_weight": 0.3},
        )

    @pytest.fixture
    def hybrid_config(self):
        """Create hybrid alignment config."""
        return AlignmentConfig(
            strategy=AlignmentStrategy.HYBRID,
            parameters={
                "length_weight": 0.4,
                "boundary_weight": 0.3,
                "semantic_weight": 0.3,
            },
        )

    @pytest.mark.asyncio
    async def test_align_translation_length_based(
        self,
        alignment_service,
        sample_timed_transcript,
        sample_translated_texts,
        length_based_config,
    ):
        """Test length-based alignment."""
        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=length_based_config,
        )

        assert isinstance(result, TimedTranslation)
        assert len(result.segments) == 3
        assert result.target_language == "es"
        assert result.alignment_strategy == AlignmentStrategy.LENGTH_BASED
        assert result.original_transcript == sample_timed_transcript

        # Check individual segments
        for i, segment in enumerate(result.segments):
            assert isinstance(segment, TimedTranslationSegment)
            assert segment.original_text == sample_timed_transcript.segments[i].text
            assert segment.translated_text == sample_translated_texts[i]
            assert segment.start_time == sample_timed_transcript.segments[i].start_time
            assert 0.0 <= segment.alignment_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_align_translation_sentence_boundary(
        self,
        alignment_service,
        sample_timed_transcript,
        sample_translated_texts,
        sentence_boundary_config,
    ):
        """Test sentence boundary alignment."""
        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=sentence_boundary_config,
        )

        assert isinstance(result, TimedTranslation)
        assert result.alignment_strategy == AlignmentStrategy.SENTENCE_BOUNDARY
        assert len(result.segments) == 3

        # Check for sentence boundary detection
        last_segment = result.segments[-1]
        assert last_segment.boundary_type == "sentence_end"

    @pytest.mark.asyncio
    async def test_align_translation_hybrid(
        self,
        alignment_service,
        sample_timed_transcript,
        sample_translated_texts,
        hybrid_config,
    ):
        """Test hybrid alignment."""
        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=hybrid_config,
        )

        assert isinstance(result, TimedTranslation)
        assert result.alignment_strategy == AlignmentStrategy.HYBRID
        assert len(result.segments) == 3

        # Hybrid should have higher confidence than individual strategies
        assert result.average_confidence > 0.0

    @pytest.mark.asyncio
    async def test_align_translation_empty_transcript(
        self, alignment_service, sample_translated_texts, length_based_config
    ):
        """Test alignment with empty transcript."""
        # Test validation directly by mocking an empty transcript
        # Create a mock transcript with empty segments
        mock_transcript = Mock()
        mock_transcript.segments = []

        with pytest.raises(AlignmentError, match="must have at least one segment"):
            await alignment_service._validate_inputs(
                timed_transcript=mock_transcript,
                translated_texts=sample_translated_texts,
                config=length_based_config,
            )

    @pytest.mark.asyncio
    async def test_align_translation_empty_translations(
        self, alignment_service, sample_timed_transcript, length_based_config
    ):
        """Test alignment with empty translations."""
        with pytest.raises(AlignmentError, match="cannot be empty"):
            await alignment_service.align_translation(
                timed_transcript=sample_timed_transcript,
                translated_texts=[],
                target_language="es",
                config=length_based_config,
            )

    @pytest.mark.asyncio
    async def test_align_translation_unsupported_strategy(
        self, alignment_service, sample_timed_transcript, sample_translated_texts
    ):
        """Test alignment with unsupported strategy."""
        config = AlignmentConfig(strategy=AlignmentStrategy.DYNAMIC_PROGRAMMING)

        with pytest.raises(AlignmentError, match="Unsupported alignment strategy"):
            await alignment_service.align_translation(
                timed_transcript=sample_timed_transcript,
                translated_texts=sample_translated_texts,
                target_language="es",
                config=config,
            )

    @pytest.mark.asyncio
    async def test_align_translation_mismatched_lengths(
        self, alignment_service, sample_timed_transcript, length_based_config
    ):
        """Test alignment with mismatched segment/translation counts."""
        # More translations than segments
        extra_translations = [
            "Hola mundo",
            "Esto es una prueba",
            "Adiós a todos",
            "Texto extra",
        ]

        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=extra_translations,
            target_language="es",
            config=length_based_config,
        )

        # Should align only the minimum count
        assert len(result.segments) == 3

    @pytest.mark.asyncio
    async def test_evaluate_alignment(
        self,
        alignment_service,
        sample_timed_transcript,
        sample_translated_texts,
        length_based_config,
    ):
        """Test alignment evaluation."""
        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=length_based_config,
        )

        evaluation = await alignment_service.evaluate_alignment(result)

        assert isinstance(evaluation, AlignmentEvaluation)
        assert evaluation.strategy == AlignmentStrategy.LENGTH_BASED
        assert 0.0 <= evaluation.timing_accuracy <= 1.0
        assert 0.0 <= evaluation.text_preservation <= 1.0
        assert 0.0 <= evaluation.boundary_alignment <= 1.0
        assert 0.0 <= evaluation.overall_score <= 1.0
        assert evaluation.execution_time > 0.0
        assert evaluation.segment_count == 3

    @pytest.mark.asyncio
    async def test_compare_alignments(
        self,
        alignment_service,
        sample_timed_transcript,
        sample_translated_texts,
        length_based_config,
        sentence_boundary_config,
    ):
        """Test alignment comparison."""
        # Create two different alignments
        alignment1 = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=length_based_config,
        )

        alignment2 = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=sentence_boundary_config,
        )

        comparisons = await alignment_service.compare_alignments(
            [alignment1, alignment2]
        )

        assert len(comparisons) == 2
        assert comparisons[0].strategy == AlignmentStrategy.LENGTH_BASED
        assert comparisons[1].strategy == AlignmentStrategy.SENTENCE_BOUNDARY

    def test_get_supported_strategies(self, alignment_service):
        """Test getting supported strategies."""
        strategies = alignment_service.get_supported_strategies()

        assert AlignmentStrategy.LENGTH_BASED in strategies
        assert AlignmentStrategy.SENTENCE_BOUNDARY in strategies
        assert AlignmentStrategy.HYBRID in strategies
        assert AlignmentStrategy.DYNAMIC_PROGRAMMING not in strategies

    @pytest.mark.asyncio
    async def test_timing_accuracy_calculation(
        self,
        alignment_service,
        sample_timed_transcript,
        sample_translated_texts,
        length_based_config,
    ):
        """Test timing accuracy calculation."""
        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=sample_translated_texts,
            target_language="es",
            config=length_based_config,
        )

        # Should have high timing accuracy since we preserve original timing
        assert result.alignment_evaluation.timing_accuracy > 0.8

    @pytest.mark.asyncio
    async def test_confidence_scoring(
        self, alignment_service, sample_timed_transcript, length_based_config
    ):
        """Test confidence scoring for different translation lengths."""
        # Test with similar length translations
        similar_translations = [
            "Hello world!",  # Similar length
            "This is a test!",  # Similar length
            "Goodbye everyone!",  # Similar length
        ]

        result1 = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=similar_translations,
            target_language="en",
            config=length_based_config,
        )

        # Test with very different length translations
        different_translations = [
            "Hi",  # Much shorter
            "This is a very long test sentence that goes on and on",  # Much longer
            "Bye",  # Much shorter
        ]

        result2 = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=different_translations,
            target_language="en",
            config=length_based_config,
        )

        # Similar lengths should have higher confidence
        assert result1.average_confidence > result2.average_confidence

    @pytest.mark.asyncio
    async def test_timing_adjustments(
        self, alignment_service, sample_timed_transcript, length_based_config
    ):
        """Test timing adjustments for different translation lengths."""
        # Test with much longer translations
        long_translations = [
            "This is a very long translation that is much longer than the original text",
            "This is another very long translation that should require timing adjustments",
            "And this is the final very long translation that will also need adjustments",
        ]

        result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=long_translations,
            target_language="en",
            config=length_based_config,
        )

        # Should have timing adjustments for longer text
        adjustments = [abs(seg.timing_adjustment) for seg in result.segments]
        assert any(adj > 0.0 for adj in adjustments)

    def test_alignment_config_defaults(self):
        """Test alignment config default parameters."""
        # Test LENGTH_BASED defaults
        config = AlignmentConfig(strategy=AlignmentStrategy.LENGTH_BASED)
        assert config.parameters["length_weight"] == 0.8
        assert config.parameters["position_weight"] == 0.2

        # Test SENTENCE_BOUNDARY defaults
        config = AlignmentConfig(strategy=AlignmentStrategy.SENTENCE_BOUNDARY)
        assert config.parameters["sentence_boundary_weight"] == 0.7
        assert config.parameters["length_weight"] == 0.3

        # Test HYBRID defaults
        config = AlignmentConfig(strategy=AlignmentStrategy.HYBRID)
        assert config.parameters["length_weight"] == 0.4
        assert config.parameters["boundary_weight"] == 0.3
        assert config.parameters["semantic_weight"] == 0.3

    def test_timed_translation_segment_validation(self):
        """Test TimedTranslationSegment validation."""
        # Valid segment
        segment = TimedTranslationSegment(
            start_time=0.0,
            end_time=10.0,
            original_text="Hello",
            translated_text="Hola",
            alignment_confidence=0.8,
        )
        assert segment.duration == 10.0

        # Invalid start time
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            TimedTranslationSegment(
                start_time=-1.0,
                end_time=10.0,
                original_text="Hello",
                translated_text="Hola",
                alignment_confidence=0.8,
            )

        # Invalid end time
        with pytest.raises(
            ValueError, match="End time must be greater than start time"
        ):
            TimedTranslationSegment(
                start_time=10.0,
                end_time=5.0,
                original_text="Hello",
                translated_text="Hola",
                alignment_confidence=0.8,
            )

        # Invalid confidence
        with pytest.raises(
            ValueError, match="Alignment confidence must be between 0.0 and 1.0"
        ):
            TimedTranslationSegment(
                start_time=0.0,
                end_time=10.0,
                original_text="Hello",
                translated_text="Hola",
                alignment_confidence=1.5,
            )

    def test_alignment_evaluation_validation(self):
        """Test AlignmentEvaluation validation."""
        # Valid evaluation
        evaluation = AlignmentEvaluation(
            strategy=AlignmentStrategy.LENGTH_BASED,
            timing_accuracy=0.8,
            text_preservation=0.9,
            boundary_alignment=0.7,
            overall_score=0.8,
            execution_time=1.5,
            segment_count=3,
            average_confidence=0.85,
        )
        assert evaluation.overall_score == 0.8

        # Invalid metric (out of range)
        with pytest.raises(
            ValueError, match="Evaluation metric must be between 0.0 and 1.0"
        ):
            AlignmentEvaluation(
                strategy=AlignmentStrategy.LENGTH_BASED,
                timing_accuracy=1.5,  # Invalid
                text_preservation=0.9,
                boundary_alignment=0.7,
                overall_score=0.8,
                execution_time=1.5,
                segment_count=3,
                average_confidence=0.85,
            )

        # Invalid execution time
        with pytest.raises(ValueError, match="Execution time cannot be negative"):
            AlignmentEvaluation(
                strategy=AlignmentStrategy.LENGTH_BASED,
                timing_accuracy=0.8,
                text_preservation=0.9,
                boundary_alignment=0.7,
                overall_score=0.8,
                execution_time=-1.0,  # Invalid
                segment_count=3,
                average_confidence=0.85,
            )
