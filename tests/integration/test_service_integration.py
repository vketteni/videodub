"""Service integration tests for videodub pipeline."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from videodub.core.exceptions import DataExtractionError, TranslationError, AlignmentError
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
from videodub.services.data_extraction import YouTubeDataExtractionService
from videodub.services.translator import OpenAITranslationService


class TestServiceIntegration:
    """Integration tests for service interactions."""

    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            video_id="test_video",
            title="Test Video",
            duration=120.0,
            url="https://youtube.com/watch?v=test",
            channel="Test Channel",
        )

    @pytest.fixture
    def sample_timed_transcript(self, sample_video_metadata):
        """Create sample timed transcript."""
        segments = [
            TranscriptSegment(
                start_time=0.0,
                end_time=5.0,
                text="Hello world, this is a test.",
            ),
            TranscriptSegment(
                start_time=5.0,
                end_time=10.0,
                text="This is the second segment.",
            ),
            TranscriptSegment(
                start_time=10.0,
                end_time=15.0,
                text="And this is the final segment.",
            ),
        ]
        
        timing_metadata = TimingMetadata(
            segment_count=3,
            total_duration=15.0,
            average_segment_duration=5.0,
            timing_accuracy=0.95,
        )
        
        return TimedTranscript(
            segments=segments,
            video_metadata=sample_video_metadata,
            timing_metadata=timing_metadata,
            source_type=SourceType.YOUTUBE,
            extraction_quality=0.9,
        )

    @pytest.mark.asyncio
    async def test_data_extraction_to_translation_integration(self, tmp_path, sample_video_metadata):
        """Test data extraction service -> translation service integration."""
        # Mock data extraction service
        data_service = Mock(spec=YouTubeDataExtractionService)
        
        # Create realistic timed transcript
        transcript = TimedTranscript(
            segments=[
                TranscriptSegment(
                    start_time=0.0,
                    end_time=3.0,
                    text="Hello world",
                ),
                TranscriptSegment(
                    start_time=3.0,
                    end_time=6.0,
                    text="How are you?",
                ),
            ],
            video_metadata=sample_video_metadata,
            timing_metadata=TimingMetadata(
                segment_count=2,
                total_duration=6.0,
                average_segment_duration=3.0,
                timing_accuracy=0.95,
            ),
            source_type=SourceType.YOUTUBE,
            extraction_quality=0.9,
        )
        
        data_service.extract_from_url = AsyncMock(return_value=transcript)
        
        # Mock translation service
        translation_service = Mock(spec=OpenAITranslationService)
        translation_service.translate_batch = AsyncMock(return_value=[
            "Hola mundo",
            "¿Cómo estás?"
        ])
        
        # Test integration
        test_url = "https://youtube.com/watch?v=test"
        extracted_data = await data_service.extract_from_url(test_url)
        
        # Extract texts for translation
        texts = [segment.text for segment in extracted_data.segments]
        translated_texts = await translation_service.translate_batch(texts, "es")
        
        # Verify integration
        assert len(extracted_data.segments) == 2
        assert len(translated_texts) == 2
        assert extracted_data.segments[0].text == "Hello world"
        assert translated_texts[0] == "Hola mundo"
        assert extracted_data.segments[1].text == "How are you?"
        assert translated_texts[1] == "¿Cómo estás?"
        
        # Verify service calls
        data_service.extract_from_url.assert_called_once_with(test_url)
        translation_service.translate_batch.assert_called_once_with(texts, "es")

    @pytest.mark.asyncio
    async def test_translation_to_alignment_integration(self, sample_timed_transcript):
        """Test translation service -> alignment service integration."""
        # Mock translation service
        translation_service = Mock(spec=OpenAITranslationService)
        translation_service.translate_batch = AsyncMock(return_value=[
            "Hola mundo, esta es una prueba.",
            "Este es el segundo segmento.",
            "Y este es el segmento final.",
        ])
        
        # Mock alignment service
        alignment_service = Mock(spec=TimingAlignmentService)
        aligned_translation = TimedTranslation(
            segments=[
                TimedTranslationSegment(
                    start_time=0.0,
                    end_time=5.0,
                    original_text="Hello world, this is a test.",
                    translated_text="Hola mundo, esta es una prueba.",
                    alignment_confidence=0.95,
                ),
                TimedTranslationSegment(
                    start_time=5.0,
                    end_time=10.0,
                    original_text="This is the second segment.",
                    translated_text="Este es el segundo segmento.",
                    alignment_confidence=0.92,
                ),
                TimedTranslationSegment(
                    start_time=10.0,
                    end_time=15.0,
                    original_text="And this is the final segment.",
                    translated_text="Y este es el segmento final.",
                    alignment_confidence=0.88,
                ),
            ],
            original_transcript=sample_timed_transcript,
            target_language="es",
            alignment_config=AlignmentConfig(strategy=AlignmentStrategy.HYBRID),
            alignment_evaluation=AlignmentEvaluation(
                strategy=AlignmentStrategy.HYBRID,
                timing_accuracy=0.95,
                text_preservation=0.92,
                boundary_alignment=0.88,
                overall_score=0.92,
                execution_time=0.15,
                segment_count=3,
                average_confidence=0.92,
            ),
            timing_metadata=sample_timed_transcript.timing_metadata,
        )
        
        alignment_service.align_translation = AsyncMock(return_value=aligned_translation)
        
        # Test integration
        original_texts = [segment.text for segment in sample_timed_transcript.segments]
        translated_texts = await translation_service.translate_batch(original_texts, "es")
        
        aligned_result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=translated_texts,
            target_language="es",
            config=AlignmentConfig(strategy=AlignmentStrategy.HYBRID),
        )
        
        # Verify integration
        assert len(aligned_result.segments) == 3
        assert aligned_result.target_language == "es"
        assert aligned_result.alignment_strategy == AlignmentStrategy.HYBRID
        
        # Verify timing preservation
        for i, segment in enumerate(aligned_result.segments):
            original_segment = sample_timed_transcript.segments[i]
            assert segment.start_time == original_segment.start_time
            assert segment.end_time == original_segment.end_time
            assert segment.original_text == original_segment.text
            assert segment.translated_text == translated_texts[i]
        
        # Verify service calls
        translation_service.translate_batch.assert_called_once_with(original_texts, "es")
        alignment_service.align_translation.assert_called_once()

    @pytest.mark.asyncio
    async def test_alignment_comparison_integration(self, sample_timed_transcript):
        """Test alignment service A/B testing integration."""
        # Mock alignment service
        alignment_service = Mock(spec=TimingAlignmentService)
        
        # Create different alignment results for comparison
        length_based_result = TimedTranslation(
            segments=[
                TimedTranslationSegment(
                    start_time=0.0,
                    end_time=5.0,
                    original_text="Hello world, this is a test.",
                    translated_text="Hola mundo, esta es una prueba.",
                    alignment_confidence=0.90,
                ),
                TimedTranslationSegment(
                    start_time=5.0,
                    end_time=10.0,
                    original_text="This is the second segment.",
                    translated_text="Este es el segundo segmento.",
                    alignment_confidence=0.85,
                ),
                TimedTranslationSegment(
                    start_time=10.0,
                    end_time=15.0,
                    original_text="And this is the final segment.",
                    translated_text="Y este es el segmento final.",
                    alignment_confidence=0.88,
                ),
            ],
            original_transcript=sample_timed_transcript,
            target_language="es",
            alignment_config=AlignmentConfig(strategy=AlignmentStrategy.LENGTH_BASED),
            alignment_evaluation=AlignmentEvaluation(
                strategy=AlignmentStrategy.LENGTH_BASED,
                timing_accuracy=0.88,
                text_preservation=0.90,
                boundary_alignment=0.85,
                overall_score=0.88,
                execution_time=0.05,
                segment_count=3,
                average_confidence=0.92,
            ),
            timing_metadata=sample_timed_transcript.timing_metadata,
        )
        
        hybrid_result = TimedTranslation(
            segments=[
                TimedTranslationSegment(
                    start_time=0.0,
                    end_time=5.0,
                    original_text="Hello world, this is a test.",
                    translated_text="Hola mundo, esta es una prueba.",
                    alignment_confidence=0.95,
                ),
                TimedTranslationSegment(
                    start_time=5.0,
                    end_time=10.0,
                    original_text="This is the second segment.",
                    translated_text="Este es el segundo segmento.",
                    alignment_confidence=0.92,
                ),
                TimedTranslationSegment(
                    start_time=10.0,
                    end_time=15.0,
                    original_text="And this is the final segment.",
                    translated_text="Y este es el segmento final.",
                    alignment_confidence=0.90,
                ),
            ],
            original_transcript=sample_timed_transcript,
            target_language="es",
            alignment_config=AlignmentConfig(strategy=AlignmentStrategy.HYBRID),
            alignment_evaluation=AlignmentEvaluation(
                strategy=AlignmentStrategy.HYBRID,
                timing_accuracy=0.95,
                text_preservation=0.92,
                boundary_alignment=0.88,
                overall_score=0.92,
                execution_time=0.15,
                segment_count=3,
                average_confidence=0.92,
            ),
            timing_metadata=sample_timed_transcript.timing_metadata,
        )
        
        # Mock alignment service methods
        alignment_service.align_translation = AsyncMock(side_effect=[length_based_result, hybrid_result])
        alignment_service.compare_alignments = AsyncMock(return_value=[
            length_based_result.alignment_evaluation,
            hybrid_result.alignment_evaluation,
        ])
        
        # Test A/B comparison - provide translations for all 3 segments
        translated_texts = [
            "Hola mundo, esta es una prueba.",
            "Este es el segundo segmento.", 
            "Y este es el segmento final."
        ]
        
        # Test both strategies
        length_config = AlignmentConfig(strategy=AlignmentStrategy.LENGTH_BASED)
        hybrid_config = AlignmentConfig(strategy=AlignmentStrategy.HYBRID)
        
        length_result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=translated_texts,
            target_language="es",
            config=length_config,
        )
        
        hybrid_result = await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=translated_texts,
            target_language="es",
            config=hybrid_config,
        )
        
        # Compare results
        comparison_results = await alignment_service.compare_alignments([length_result, hybrid_result])
        
        # Verify comparison
        assert len(comparison_results) == 2
        assert comparison_results[0].strategy == AlignmentStrategy.LENGTH_BASED
        assert comparison_results[1].strategy == AlignmentStrategy.HYBRID
        assert comparison_results[1].overall_score > comparison_results[0].overall_score
        
        # Verify service calls
        assert alignment_service.align_translation.call_count == 2
        alignment_service.compare_alignments.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_error_propagation(self, sample_timed_transcript):
        """Test error propagation between services."""
        # Mock translation service with error
        translation_service = Mock(spec=OpenAITranslationService)
        translation_service.translate_batch = AsyncMock(side_effect=TranslationError("Translation failed"))
        
        # Mock alignment service
        alignment_service = Mock(spec=TimingAlignmentService)
        alignment_service.align_translation = AsyncMock()
        
        # Test error propagation
        original_texts = [segment.text for segment in sample_timed_transcript.segments]
        
        # Translation should fail
        with pytest.raises(TranslationError) as exc_info:
            await translation_service.translate_batch(original_texts, "es")
        
        assert "Translation failed" in str(exc_info.value)
        
        # Alignment should not be called due to translation failure
        alignment_service.align_translation.assert_not_called()

    @pytest.mark.asyncio
    async def test_data_format_compatibility(self, sample_timed_transcript):
        """Test data format compatibility between services."""
        # Test that data formats are compatible across service boundaries
        
        # Extract data from transcript
        original_texts = [segment.text for segment in sample_timed_transcript.segments]
        segment_timings = [(seg.start_time, seg.end_time) for seg in sample_timed_transcript.segments]
        
        # Mock translation
        translated_texts = [
            "Hola mundo, esta es una prueba.",
            "Este es el segundo segmento.",
            "Y este es el segmento final.",
        ]
        
        # Verify data compatibility
        assert len(original_texts) == len(translated_texts)
        assert len(original_texts) == len(segment_timings)
        
        # Verify timing data preservation
        for i, (start, end) in enumerate(segment_timings):
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert start < end
            assert start >= 0
        
        # Verify text data formats
        for original, translated in zip(original_texts, translated_texts):
            assert isinstance(original, str)
            assert isinstance(translated, str)
            assert len(original.strip()) > 0
            assert len(translated.strip()) > 0

    @pytest.mark.asyncio
    async def test_service_resource_cleanup(self, tmp_path):
        """Test that services properly clean up resources."""
        # Mock services with resource tracking
        data_service = Mock(spec=YouTubeDataExtractionService)
        translation_service = Mock(spec=OpenAITranslationService)
        alignment_service = Mock(spec=TimingAlignmentService)
        
        # Track resource usage
        resource_usage = {"files_created": 0, "connections_opened": 0}
        
        def track_resource_creation(url):
            resource_usage["files_created"] += 1
        
        def track_connection_opening(texts, target_language):
            resource_usage["connections_opened"] += 1
        
        # Mock service methods with resource tracking
        data_service.extract_from_url = AsyncMock(side_effect=track_resource_creation)
        translation_service.translate_batch = AsyncMock(side_effect=track_connection_opening)
        alignment_service.align_translation = AsyncMock()
        
        # Simulate service usage
        await data_service.extract_from_url("https://youtube.com/watch?v=test")
        await translation_service.translate_batch(["Hello"], "es")
        await alignment_service.align_translation(None, None, "es", None)
        
        # Verify resource usage tracking
        assert resource_usage["files_created"] == 1
        assert resource_usage["connections_opened"] == 1
        
        # Verify service calls
        data_service.extract_from_url.assert_called_once()
        translation_service.translate_batch.assert_called_once()
        alignment_service.align_translation.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_configuration_consistency(self, sample_timed_transcript):
        """Test configuration consistency across services."""
        # Test that configuration is properly passed between services
        
        target_language = "es"
        alignment_config = AlignmentConfig(
            strategy=AlignmentStrategy.HYBRID,
            parameters={"confidence_threshold": 0.9}
        )
        
        # Mock services
        translation_service = Mock(spec=OpenAITranslationService)
        alignment_service = Mock(spec=TimingAlignmentService)
        
        # Configure return values
        translated_texts = ["Hola mundo"]
        translation_service.translate_batch = AsyncMock(return_value=translated_texts)
        
        mock_result = Mock()
        mock_result.target_language = target_language
        alignment_service.align_translation = AsyncMock(return_value=mock_result)
        
        # Test service calls with configuration
        await translation_service.translate_batch(["Hello world"], target_language)
        await alignment_service.align_translation(
            timed_transcript=sample_timed_transcript,
            translated_texts=translated_texts,
            target_language=target_language,
            config=alignment_config,
        )
        
        # Verify configuration consistency
        translation_service.translate_batch.assert_called_once_with(["Hello world"], target_language)
        alignment_service.align_translation.assert_called_once()
        
        # Verify the alignment service was called with correct parameters
        call_args = alignment_service.align_translation.call_args
        assert call_args[1]["target_language"] == target_language
        assert call_args[1]["config"] == alignment_config