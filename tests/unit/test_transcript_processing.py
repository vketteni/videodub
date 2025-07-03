"""Unit tests for transcript processing service."""

from unittest.mock import AsyncMock, patch

import pytest
from videodub.core.models import (
    ProcessedSegment,
    ProcessingMode,
    TranscriptSegment,
)
from videodub.services.transcript import (
    HybridTranscriptProcessingService,
    ProcessingConfig,
)


@pytest.fixture
def processing_config():
    """Default processing configuration for tests."""
    return ProcessingConfig(
        min_segment_duration=0.5,
        max_segment_duration=30.0,
        merge_threshold_seconds=2.0,
        min_words_per_segment=2,
    )


@pytest.fixture
def transcript_service(processing_config):
    """Transcript processing service instance."""
    return HybridTranscriptProcessingService(processing_config)


@pytest.fixture
def simple_segments():
    """Simple transcript segments for basic testing."""
    return [
        TranscriptSegment(start_time=0.0, end_time=3.0, text="Hello world"),
        TranscriptSegment(start_time=3.0, end_time=6.0, text="This is a test"),
        TranscriptSegment(start_time=6.0, end_time=9.0, text="Thank you")
    ]


@pytest.fixture
def fragmented_segments():
    """Fragmented segments that need merging and processing."""
    return [
        TranscriptSegment(start_time=0.0, end_time=1.0, text="Hello"),
        TranscriptSegment(start_time=1.0, end_time=2.0, text="this is"),
        TranscriptSegment(start_time=2.5, end_time=4.0, text="a fragmented sentence"),
        TranscriptSegment(start_time=5.0, end_time=7.0, text="Another complete sentence."),
        TranscriptSegment(start_time=8.0, end_time=9.0, text="End"),
    ]


@pytest.fixture
def problematic_segments():
    """Segments with various quality issues."""
    return [
        TranscriptSegment(start_time=0.0, end_time=0.3, text="uh"),  # Too short
        TranscriptSegment(start_time=1.0, end_time=2.0, text="hello    world"),  # Extra whitespace
        TranscriptSegment(start_time=2.0, end_time=3.0, text="no punctuation"),
        TranscriptSegment(start_time=3.0, end_time=4.0, text="UPPERCASE TEXT"),
        TranscriptSegment(start_time=4.0, end_time=5.0, text="mixed Case text"),
        TranscriptSegment(start_time=10.0, end_time=11.0, text="Big gap before"),
    ]


@pytest.fixture
def overlapping_segments():
    """Segments with timing overlaps."""
    return [
        TranscriptSegment(start_time=0.0, end_time=3.0, text="First segment"),
        TranscriptSegment(start_time=2.5, end_time=5.0, text="Overlapping segment"),
        TranscriptSegment(start_time=5.0, end_time=8.0, text="Third segment"),
    ]


@pytest.fixture
def empty_and_invalid_segments():
    """Segments with empty or invalid content."""
    return [
        TranscriptSegment(start_time=0.0, end_time=1.0, text="Valid text"),
        TranscriptSegment(start_time=2.0, end_time=3.0, text="   "),  # Whitespace only
        TranscriptSegment(start_time=4.0, end_time=5.0, text="Another valid"),
    ]


class TestProcessedSegmentModel:
    """Test ProcessedSegment model validation and properties."""

    def test_create_processed_segment(self, simple_segments):
        """Test creating a valid ProcessedSegment."""
        processed = ProcessedSegment(
            merged_segments=[simple_segments[0]],
            processed_text="Hello world.",
            processing_mode=ProcessingMode.RULE_BASED,
            sequence_number=0,
            original_indices=[0],
            is_sentence_complete=True,
            context_quality_score=0.8,
        )

        assert processed.merged_segments == [simple_segments[0]]
        assert processed.processed_text == "Hello world."
        assert processed.processing_mode == ProcessingMode.RULE_BASED
        assert processed.sequence_number == 0
        assert processed.original_indices == [0]
        assert processed.is_sentence_complete is True
        assert processed.context_quality_score == 0.8
        assert processed.timing_preserved is True
        assert processed.ready_for_translation is True

    def test_processed_segment_sequence_tracking(self, simple_segments):
        """Test sequence number and original indices tracking."""
        # Test single segment
        processed_single = ProcessedSegment(
            merged_segments=[simple_segments[0]],
            processed_text="Hello world.",
            processing_mode=ProcessingMode.RULE_BASED,
            sequence_number=0,
            original_indices=[0],
        )
        
        assert processed_single.sequence_number == 0
        assert processed_single.original_indices == [0]
        
        # Test merged segments
        processed_merged = ProcessedSegment(
            merged_segments=simple_segments[:2],
            processed_text="Hello world. This is a test.",
            processing_mode=ProcessingMode.RULE_BASED,
            sequence_number=1,
            original_indices=[0, 1],
        )
        
        assert processed_merged.sequence_number == 1
        assert processed_merged.original_indices == [0, 1]
        
        # Test auto-population of original_indices when not provided
        processed_auto = ProcessedSegment(
            merged_segments=simple_segments[:2],
            processed_text="Auto indices.",
            processing_mode=ProcessingMode.RULE_BASED,
            sequence_number=2,
        )
        
        # Should auto-populate as [0, 1] for 2 segments
        assert processed_auto.original_indices == [0, 1]

    def test_processed_segment_timing_properties(self, simple_segments):
        """Test timing properties of ProcessedSegment."""
        # Single segment
        processed = ProcessedSegment(
            merged_segments=[simple_segments[0]],
            processed_text="Hello world.",
            processing_mode=ProcessingMode.RULE_BASED,
            sequence_number=0,
        )

        assert processed.start_time == 0.0
        assert processed.end_time == 3.0
        assert processed.duration == 3.0

        # Multiple segments
        processed_multi = ProcessedSegment(
            merged_segments=simple_segments,
            processed_text="Hello world. This is a test. Thank you.",
            processing_mode=ProcessingMode.RULE_BASED,
            sequence_number=1,
        )

        assert processed_multi.start_time == 0.0
        assert processed_multi.end_time == 9.0
        assert processed_multi.duration == 9.0

    def test_processed_segment_validation(self, simple_segments):
        """Test ProcessedSegment validation."""
        # Empty processed text
        with pytest.raises(ValueError, match="Processed text cannot be empty"):
            ProcessedSegment(
                merged_segments=[simple_segments[0]],
                processed_text="",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=0,
            )

        # Whitespace only text
        with pytest.raises(ValueError, match="Processed text cannot be empty"):
            ProcessedSegment(
                merged_segments=[simple_segments[0]],
                processed_text="   ",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=0,
            )

        # Empty merged segments
        with pytest.raises(ValueError, match="Must have at least one merged segment"):
            ProcessedSegment(
                merged_segments=[],
                processed_text="Some text",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=0,
            )

        # Negative sequence number
        with pytest.raises(ValueError, match="Sequence number must be non-negative"):
            ProcessedSegment(
                merged_segments=[simple_segments[0]],
                processed_text="Some text",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=-1,
            )

        # Invalid quality score
        with pytest.raises(ValueError, match="Context quality score must be between 0.0 and 1.0"):
            ProcessedSegment(
                merged_segments=[simple_segments[0]],
                processed_text="Some text",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=0,
                context_quality_score=1.5,
            )

        with pytest.raises(ValueError, match="Context quality score must be between 0.0 and 1.0"):
            ProcessedSegment(
                merged_segments=[simple_segments[0]],
                processed_text="Some text",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=0,
                context_quality_score=-0.1,
            )


class TestTranscriptProcessingValidation:
    """Test transcript segment validation."""

    @pytest.mark.asyncio
    async def test_validate_segments_success(self, transcript_service, simple_segments):
        """Test successful segment validation."""
        result = await transcript_service.validate_segments(simple_segments)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_empty_segments(self, transcript_service):
        """Test validation of empty segment list."""
        result = await transcript_service.validate_segments([])
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_segments_with_invalid_timing(self, transcript_service):
        """Test validation with invalid timing."""
        # This should fail during TranscriptSegment creation itself
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            TranscriptSegment(start_time=-1.0, end_time=3.0, text="Invalid start")

    @pytest.mark.asyncio
    async def test_validate_overlapping_segments(self, transcript_service, overlapping_segments):
        """Test validation with overlapping segments (should warn but pass)."""
        with pytest.warns(None) as warning_list:
            result = await transcript_service.validate_segments(overlapping_segments)
            assert result is True

        # Check if warning was logged (implementation detail)


class TestSequenceOrdering:
    """Test sequence ordering and parallel processing safety."""

    @pytest.mark.asyncio
    async def test_sequence_number_assignment(self, transcript_service):
        """Test that sequence numbers are assigned correctly."""
        segments = [
            TranscriptSegment(start_time=0.0, end_time=2.0, text="First segment"),
            TranscriptSegment(start_time=2.0, end_time=4.0, text="Second segment"),
            TranscriptSegment(start_time=4.0, end_time=6.0, text="Third segment"),
        ]

        result = await transcript_service.process_transcript(segments)

        # Sequence numbers should be assigned in order
        sequence_numbers = [processed.sequence_number for processed in result]
        assert sequence_numbers == sorted(sequence_numbers), "Sequence numbers should be in order"
        assert sequence_numbers == list(range(len(result))), "Sequence numbers should start from 0"

    @pytest.mark.asyncio
    async def test_original_indices_tracking(self, transcript_service):
        """Test that original indices are tracked correctly."""
        segments = [
            TranscriptSegment(start_time=0.0, end_time=1.0, text="One"),
            TranscriptSegment(start_time=1.0, end_time=2.0, text="Two"),
            TranscriptSegment(start_time=2.0, end_time=3.0, text="Three"),
            TranscriptSegment(start_time=10.0, end_time=11.0, text="Four after gap"),
        ]

        result = await transcript_service.process_transcript(segments)

        # Verify that all original indices are accounted for
        all_original_indices = []
        for processed in result:
            all_original_indices.extend(processed.original_indices)

        # Should include all original segment indices
        assert set(all_original_indices) == {0, 1, 2, 3}

    @pytest.mark.asyncio
    async def test_deterministic_ordering_after_merging(self, transcript_service):
        """Test that ordering is deterministic even after segment merging."""
        # Create segments that will be merged differently
        segments = [
            TranscriptSegment(start_time=0.0, end_time=1.0, text="Short"),
            TranscriptSegment(start_time=1.0, end_time=2.0, text="also short"),
            TranscriptSegment(start_time=2.0, end_time=5.0, text="This is a longer segment that should not be merged"),
            TranscriptSegment(start_time=10.0, end_time=11.0, text="Gap"),
            TranscriptSegment(start_time=11.0, end_time=12.0, text="after"),
        ]

        result = await transcript_service.process_transcript(segments)

        # Results should be ordered by sequence number
        for i in range(len(result) - 1):
            assert result[i].sequence_number < result[i + 1].sequence_number

        # And by start time
        for i in range(len(result) - 1):
            assert result[i].start_time <= result[i + 1].start_time

    @pytest.mark.asyncio
    async def test_parallel_processing_safety(self, transcript_service):
        """Test that results can be safely reordered after parallel processing."""
        segments = [
            TranscriptSegment(start_time=0.0, end_time=2.0, text="First"),
            TranscriptSegment(start_time=2.0, end_time=4.0, text="Second"),
            TranscriptSegment(start_time=4.0, end_time=6.0, text="Third"),
        ]

        result = await transcript_service.process_transcript(segments)

        # Simulate parallel processing by shuffling results
        import random
        shuffled_result = result.copy()
        random.shuffle(shuffled_result)

        # Should be able to restore original order using sequence_number
        restored_order = sorted(shuffled_result, key=lambda x: x.sequence_number)

        # Verify order is restored correctly
        assert [p.sequence_number for p in restored_order] == [p.sequence_number for p in result]
        assert [p.start_time for p in restored_order] == [p.start_time for p in result]

    def test_sort_processed_segments_utility(self):
        """Test utility function for sorting processed segments."""
        from videodub.core.models import ProcessedSegment, ProcessingMode, TranscriptSegment
        
        # Create segments in wrong order
        segments = [
            ProcessedSegment(
                merged_segments=[TranscriptSegment(start_time=4.0, end_time=6.0, text="Third")],
                processed_text="Third.",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=2,
                original_indices=[2],
            ),
            ProcessedSegment(
                merged_segments=[TranscriptSegment(start_time=0.0, end_time=2.0, text="First")],
                processed_text="First.",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=0,
                original_indices=[0],
            ),
            ProcessedSegment(
                merged_segments=[TranscriptSegment(start_time=2.0, end_time=4.0, text="Second")],
                processed_text="Second.",
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=1,
                original_indices=[1],
            ),
        ]
        
        # Sort by sequence number
        sorted_segments = sorted(segments, key=lambda x: x.sequence_number)
        
        # Verify correct order
        assert [s.sequence_number for s in sorted_segments] == [0, 1, 2]
        assert [s.processed_text for s in sorted_segments] == ["First.", "Second.", "Third."]
        assert [s.start_time for s in sorted_segments] == [0.0, 2.0, 4.0]


class TestRuleBasedProcessing:
    """Test rule-based preprocessing functionality."""

    @pytest.mark.asyncio
    async def test_process_simple_segments_rule_based(self, transcript_service, simple_segments):
        """Test processing simple segments with rule-based mode."""
        result = await transcript_service.process_transcript(
            simple_segments,
            mode=ProcessingMode.RULE_BASED
        )

        # Should produce some results (may merge adjacent segments)
        assert len(result) > 0
        assert len(result) <= len(simple_segments)

        for processed in result:
            assert isinstance(processed, ProcessedSegment)
            assert processed.processing_mode == ProcessingMode.RULE_BASED
            assert len(processed.merged_segments) >= 1
            assert processed.ready_for_translation is True

    @pytest.mark.asyncio
    async def test_merge_fragmented_segments(self, transcript_service, fragmented_segments):
        """Test merging of fragmented segments."""
        result = await transcript_service.process_transcript(
            fragmented_segments,
            mode=ProcessingMode.RULE_BASED
        )

        # Should merge some segments
        assert len(result) < len(fragmented_segments)

        # At least one processed segment should contain multiple merged segments
        merged_found = any(len(processed.merged_segments) > 1 for processed in result)
        assert merged_found, "Expected at least one segment to be merged from multiple inputs"

        # Find a merged segment and verify its properties
        merged_segment = next(processed for processed in result if len(processed.merged_segments) > 1)
        assert merged_segment.start_time == merged_segment.merged_segments[0].start_time
        assert merged_segment.end_time == merged_segment.merged_segments[-1].end_time

    @pytest.mark.asyncio
    async def test_punctuation_and_capitalization_fixes(self, transcript_service):
        """Test punctuation and capitalization fixes."""
        segments = [
            TranscriptSegment(start_time=0.0, end_time=3.0, text="hello world"),
            TranscriptSegment(start_time=4.0, end_time=7.0, text="another sentence"),
        ]

        result = await transcript_service.process_transcript(
            segments,
            mode=ProcessingMode.RULE_BASED
        )

        # Check that text is properly capitalized and punctuated
        for processed in result:
            text = processed.processed_text
            assert text[0].isupper(), f"Text should start with capital: {text}"
            assert text.endswith('.') or text.endswith('!') or text.endswith('?'), \
                f"Text should end with punctuation: {text}"

    @pytest.mark.asyncio
    async def test_filter_short_segments(self, transcript_service, problematic_segments):
        """Test filtering of very short segments."""
        result = await transcript_service.process_transcript(
            problematic_segments,
            mode=ProcessingMode.RULE_BASED
        )

        # Should filter out segments that are too short or have too few words
        assert len(result) < len(problematic_segments)

        # Check that all remaining segments meet minimum requirements
        for processed in result:
            assert processed.duration >= transcript_service.config.min_segment_duration

    @pytest.mark.asyncio
    async def test_whitespace_normalization(self, transcript_service):
        """Test normalization of whitespace."""
        segments = [
            TranscriptSegment(start_time=0.0, end_time=3.0, text="hello    world   test"),
        ]

        result = await transcript_service.process_transcript(
            segments,
            mode=ProcessingMode.RULE_BASED
        )

        processed_text = result[0].processed_text
        assert "  " not in processed_text  # No double spaces
        assert processed_text.strip() == processed_text  # No leading/trailing spaces


class TestQualityScoring:
    """Test quality score calculation."""

    def test_quality_score_complete_sentence(self, transcript_service):
        """Test quality score for complete sentences."""
        segments = [TranscriptSegment(start_time=0.0, end_time=3.0, text="Hello world")]
        score = transcript_service._calculate_quality_score("Hello world.", segments)

        assert 0.0 <= score <= 1.0
        # Should get bonus for sentence completeness, capitalization, punctuation
        assert score > 0.5

    def test_quality_score_incomplete_sentence(self, transcript_service):
        """Test quality score for incomplete sentences."""
        segments = [TranscriptSegment(start_time=0.0, end_time=3.0, text="hello world")]
        score = transcript_service._calculate_quality_score("hello world", segments)

        assert 0.0 <= score <= 1.0
        # Should get lower score for missing capitalization and punctuation
        assert score < 0.8

    def test_quality_score_optimal_length(self, transcript_service):
        """Test quality score with optimal text length."""
        segments = [TranscriptSegment(start_time=0.0, end_time=5.0, text="text")]

        # Optimal length (5-20 words)
        optimal_text = "This is a sentence with optimal length for translation."
        score_optimal = transcript_service._calculate_quality_score(optimal_text, segments)

        # Too short
        short_text = "Hi."
        score_short = transcript_service._calculate_quality_score(short_text, segments)

        # Too long
        long_text = " ".join(["word"] * 35)
        score_long = transcript_service._calculate_quality_score(long_text, segments)

        assert score_optimal > score_short
        assert score_optimal > score_long

    def test_sentence_completeness_detection(self, transcript_service):
        """Test sentence completeness detection."""
        # Complete sentences with punctuation
        assert transcript_service._is_sentence_complete("Hello world.") is True
        assert transcript_service._is_sentence_complete("What's up?") is True
        assert transcript_service._is_sentence_complete("Amazing!") is True

        # Longer phrases without punctuation (3+ words)
        assert transcript_service._is_sentence_complete("hello world everyone") is True
        assert transcript_service._is_sentence_complete("this is working") is True

        # Short incomplete phrases
        assert transcript_service._is_sentence_complete("hello world") is False  # Only 2 words
        assert transcript_service._is_sentence_complete("and then") is False
        assert transcript_service._is_sentence_complete("because") is False
        assert transcript_service._is_sentence_complete("the") is False
        assert transcript_service._is_sentence_complete("is") is False
        assert transcript_service._is_sentence_complete("") is False


class TestProcessingModes:
    """Test different processing modes."""

    @pytest.mark.asyncio
    async def test_rule_based_mode(self, transcript_service, simple_segments):
        """Test rule-based only processing mode."""
        result = await transcript_service.process_transcript(
            simple_segments,
            mode=ProcessingMode.RULE_BASED
        )

        for processed in result:
            assert processed.processing_mode == ProcessingMode.RULE_BASED
            assert "ai_enhanced" not in processed.enhancement_metadata

    @pytest.mark.asyncio
    async def test_ai_enhanced_mode(self, transcript_service, simple_segments):
        """Test AI-enhanced processing mode."""
        with patch.object(transcript_service, '_placeholder_ai_enhancement', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = "AI enhanced text."

            result = await transcript_service.process_transcript(
                simple_segments,
                mode=ProcessingMode.AI_ENHANCED
            )

            # AI enhancement should be called for all segments
            assert mock_ai.call_count == len(result)

            for processed in result:
                assert processed.processing_mode == ProcessingMode.AI_ENHANCED
                assert processed.enhancement_metadata.get("ai_enhanced") is True

    @pytest.mark.asyncio
    async def test_hybrid_mode_with_high_quality(self, transcript_service):
        """Test hybrid mode skipping AI enhancement for high-quality segments."""
        # Create segments that will have high quality scores
        high_quality_segments = [
            TranscriptSegment(start_time=0.0, end_time=5.0, text="This is a perfect sentence."),
        ]

        with patch.object(transcript_service, '_calculate_quality_score', return_value=0.9):
            with patch.object(transcript_service, '_placeholder_ai_enhancement', new_callable=AsyncMock) as mock_ai:
                result = await transcript_service.process_transcript(
                    high_quality_segments,
                    mode=ProcessingMode.HYBRID
                )

                # AI enhancement should be skipped for high-quality segments
                mock_ai.assert_not_called()

                for processed in result:
                    assert processed.processing_mode == ProcessingMode.RULE_BASED

    @pytest.mark.asyncio
    async def test_hybrid_mode_with_low_quality(self, transcript_service):
        """Test hybrid mode applying AI enhancement for low-quality segments."""
        low_quality_segments = [
            TranscriptSegment(start_time=0.0, end_time=2.0, text="uh hello"),
        ]

        with patch.object(transcript_service, '_calculate_quality_score', return_value=0.3):
            with patch.object(transcript_service, '_placeholder_ai_enhancement', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = "Enhanced hello."

                result = await transcript_service.process_transcript(
                    low_quality_segments,
                    mode=ProcessingMode.HYBRID
                )

                # AI enhancement should be called for low-quality segments
                assert mock_ai.call_count == len(result)

                for processed in result:
                    assert processed.processing_mode == ProcessingMode.HYBRID
                    assert processed.enhancement_metadata.get("ai_enhanced") is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_segment_list(self, transcript_service):
        """Test processing empty segment list."""
        result = await transcript_service.process_transcript([])
        assert result == []

    @pytest.mark.asyncio
    async def test_all_segments_filtered_out(self, transcript_service):
        """Test when segments are very low quality but still get merged."""
        very_short_segments = [
            TranscriptSegment(start_time=0.0, end_time=0.1, text="a"),
            TranscriptSegment(start_time=1.0, end_time=1.1, text="b"),
        ]

        result = await transcript_service.process_transcript(very_short_segments)
        # Should merge low-quality segments rather than drop them entirely
        assert len(result) == 1
        assert len(result[0].merged_segments) == 2
        assert "A b" in result[0].processed_text

    @pytest.mark.asyncio
    async def test_single_word_segments(self, transcript_service):
        """Test processing segments with single words."""
        single_word_segments = [
            TranscriptSegment(start_time=0.0, end_time=1.0, text="Hello"),
            TranscriptSegment(start_time=1.0, end_time=2.0, text="there"),
            TranscriptSegment(start_time=2.0, end_time=3.0, text="friend"),
        ]

        result = await transcript_service.process_transcript(single_word_segments)

        # These should be merged into fewer segments
        assert len(result) <= len(single_word_segments)
        if result:
            # Merged text should be properly formatted
            merged_text = result[0].processed_text
            assert merged_text[0].isupper()
            assert merged_text.endswith('.')

    @pytest.mark.asyncio
    async def test_very_long_segments(self, transcript_service):
        """Test processing very long segments."""
        long_text = " ".join(["word"] * 100)  # Very long segment
        long_segments = [
            TranscriptSegment(start_time=0.0, end_time=60.0, text=long_text),
        ]

        result = await transcript_service.process_transcript(long_segments)

        # Should still process but may get lower quality score
        assert len(result) == 1
        assert len(result[0].processed_text) > 100

    @pytest.mark.asyncio
    async def test_gaps_in_timing(self, transcript_service):
        """Test processing segments with large timing gaps."""
        gapped_segments = [
            TranscriptSegment(start_time=0.0, end_time=2.0, text="First segment"),
            TranscriptSegment(start_time=10.0, end_time=12.0, text="Second segment after gap"),
            TranscriptSegment(start_time=30.0, end_time=32.0, text="Third segment after bigger gap"),
        ]

        result = await transcript_service.process_transcript(gapped_segments)

        # Segments with large gaps should not be merged
        assert len(result) == 3
        for i, processed in enumerate(result):
            assert len(processed.merged_segments) == 1
            assert processed.merged_segments[0] == gapped_segments[i]


class TestIntegrationWorkflow:
    """Test complete processing workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_processing_workflow(self, transcript_service, fragmented_segments):
        """Test complete processing workflow from fragmented input to translation-ready output."""
        # Process with hybrid mode
        result = await transcript_service.process_transcript(
            fragmented_segments,
            mode=ProcessingMode.HYBRID
        )

        # Verify all results are translation-ready
        assert all(processed.ready_for_translation for processed in result)

        # Verify timing preservation (allowing for filtering of very short segments)
        original_start = min(seg.start_time for seg in fragmented_segments)
        original_end = max(seg.end_time for seg in fragmented_segments)

        if result:  # Only check if we have results
            processed_start = min(processed.start_time for processed in result)
            processed_end = max(processed.end_time for processed in result)

            # Start time should be >= original start (due to possible filtering)
            assert processed_start >= original_start
            # End time should be preserved exactly (max of all segments)
            assert processed_end == original_end

        # Verify enhancement metadata is present
        for processed in result:
            assert "rule_based_processing" in processed.enhancement_metadata
            assert processed.enhancement_metadata["segments_merged"] >= 1

    @pytest.mark.asyncio
    async def test_processing_preserves_content(self, transcript_service, simple_segments):
        """Test that processing preserves essential content."""
        original_text = " ".join(seg.text for seg in simple_segments)

        result = await transcript_service.process_transcript(simple_segments)

        processed_text = " ".join(processed.processed_text for processed in result)

        # Essential words should be preserved (allowing for punctuation/capitalization changes)
        original_words = set(original_text.lower().split())
        processed_words = set(processed_text.lower().replace('.', '').replace('!', '').replace('?', '').split())

        # Most words should be preserved
        preserved_ratio = len(original_words.intersection(processed_words)) / len(original_words)
        assert preserved_ratio > 0.8, f"Too many words lost: {original_words} vs {processed_words}"

    @pytest.mark.asyncio
    async def test_custom_processing_config(self):
        """Test processing with custom configuration."""
        custom_config = ProcessingConfig(
            min_segment_duration=1.0,  # Higher threshold
            merge_threshold_seconds=0.5,  # Lower threshold
            min_words_per_segment=3,  # Higher word requirement
        )

        service = HybridTranscriptProcessingService(custom_config)

        segments = [
            TranscriptSegment(start_time=0.0, end_time=0.8, text="short"),  # Below duration threshold
            TranscriptSegment(start_time=1.0, end_time=2.0, text="hello world"),  # Below word threshold
            TranscriptSegment(start_time=3.0, end_time=5.0, text="this should be kept"),
        ]

        result = await service.process_transcript(segments)

        # Should filter out segments not meeting criteria
        assert len(result) <= len(segments)
        if result:
            for processed in result:
                assert processed.duration >= custom_config.min_segment_duration
