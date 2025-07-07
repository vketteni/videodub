"""End-to-end pipeline integration tests."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from videodub.core.exceptions import DataExtractionError, TranslationError, AlignmentError
from videodub.core.interfaces import (
    AlignmentService,
    AudioProcessingService,
    DataExtractionService,
    StorageService,
    TranslationService,
    TTSService,
    VideoProcessingService,
)
from videodub.core.models import (
    AlignmentConfig,
    AlignmentEvaluation,
    AlignmentStrategy,
    PipelineConfig,
    ProcessingStatus,
    SourceType,
    TimedTranscript,
    TimedTranslation,
    TimedTranslationSegment,
    TimingMetadata,
    TranscriptSegment,
    TTSEngine,
    VideoMetadata,
)
from videodub.core.pipeline import TranslationPipeline


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

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
            source_type=SourceType.YOUTUBE,
            timing_metadata=timing_metadata,
            video_metadata=sample_video_metadata,
            extraction_quality=0.9,
        )

    @pytest.fixture
    def sample_timed_translation(self, sample_video_metadata, sample_timed_transcript):
        """Create sample timed translation."""
        segments = [
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
        ]
        
        alignment_config = AlignmentConfig(strategy=AlignmentStrategy.HYBRID)
        
        evaluation = AlignmentEvaluation(
            strategy=AlignmentStrategy.HYBRID,
            timing_accuracy=0.95,
            text_preservation=0.92,
            boundary_alignment=0.88,
            overall_score=0.92,
            execution_time=0.15,
            segment_count=3,
            average_confidence=0.92,
        )
        
        timing_metadata = TimingMetadata(
            segment_count=3,
            total_duration=15.0,
            average_segment_duration=5.0,
            timing_accuracy=0.95,
        )
        
        return TimedTranslation(
            segments=segments,
            original_transcript=sample_timed_transcript,
            target_language="es",
            alignment_config=alignment_config,
            alignment_evaluation=evaluation,
            timing_metadata=timing_metadata,
        )

    @pytest.fixture
    def mock_services(self, tmp_path, sample_timed_transcript, sample_timed_translation):
        """Create mock services with realistic behaviors."""
        # Mock DataExtractionService
        data_extraction_service = Mock(spec=DataExtractionService)
        data_extraction_service.extract_from_url = AsyncMock(return_value=sample_timed_transcript)
        
        # Mock TranslationService
        translation_service = Mock(spec=TranslationService)
        translation_service.translate_batch = AsyncMock(return_value=[
            "Hola mundo, esta es una prueba.",
            "Este es el segundo segmento.",
            "Y este es el segmento final.",
        ])
        
        # Mock AlignmentService
        alignment_service = Mock(spec=AlignmentService)
        alignment_service.align_translation = AsyncMock(return_value=sample_timed_translation)
        alignment_service.compare_alignments = AsyncMock(return_value=[
            sample_timed_translation.alignment_evaluation
        ])
        
        # Mock TTSService
        tts_service = Mock(spec=TTSService)
        audio_files = [tmp_path / f"audio_{i}.wav" for i in range(3)]
        for audio_file in audio_files:
            audio_file.write_text("fake audio")
        
        async def mock_generate_batch_audio(segments, output_directory, language, voice=None):
            for audio_file in audio_files:
                yield audio_file
        
        tts_service.generate_batch_audio = mock_generate_batch_audio
        
        # Mock AudioProcessingService
        audio_service = Mock(spec=AudioProcessingService)
        audio_service.combine_audio_segments = AsyncMock(return_value=tmp_path / "combined.wav")
        
        # Mock VideoProcessingService
        video_service = Mock(spec=VideoProcessingService)
        video_service.create_dubbed_video = AsyncMock(return_value=tmp_path / "dubbed.mp4")
        
        # Mock StorageService
        storage_service = Mock(spec=StorageService)
        storage_service.get_video_directory = AsyncMock(return_value=tmp_path)
        storage_service.save_metadata = AsyncMock(return_value=tmp_path / "metadata.json")
        storage_service.save_timed_transcript = AsyncMock(return_value=tmp_path / "transcript.json")
        storage_service.save_timed_translation = AsyncMock(return_value=tmp_path / "translation.json")
        storage_service.save_processing_result = AsyncMock()
        
        return {
            "data_extraction": data_extraction_service,
            "translation": translation_service,
            "alignment": alignment_service,
            "tts": tts_service,
            "audio": audio_service,
            "video": video_service,
            "storage": storage_service,
        }

    @pytest.fixture
    def pipeline_config(self, tmp_path):
        """Create pipeline configuration."""
        return PipelineConfig(
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
            output_directory=tmp_path,
        )

    @pytest.fixture
    def pipeline(self, mock_services, pipeline_config):
        """Create TranslationPipeline with mocked services."""
        return TranslationPipeline(
            data_extraction_service=mock_services["data_extraction"],
            translation_service=mock_services["translation"],
            alignment_service=mock_services["alignment"],
            tts_service=mock_services["tts"],
            audio_service=mock_services["audio"],
            video_processing_service=mock_services["video"],
            storage_service=mock_services["storage"],
            config=pipeline_config,
        )

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, pipeline, mock_services):
        """Test complete pipeline from URL to final output."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Execute the pipeline
        result = await pipeline.process_video(test_url)
        
        # Verify the result
        assert result.status == ProcessingStatus.COMPLETED
        assert result.video_id == "test"
        assert result.target_language == "es"
        assert "metadata" in result.files
        assert "timed_transcript" in result.files
        assert "timed_translation" in result.files
        assert "alignment_evaluation" in result.files
        
        # Verify service calls
        mock_services["data_extraction"].extract_from_url.assert_called_once_with(test_url)
        mock_services["translation"].translate_batch.assert_called_once()
        mock_services["alignment"].align_translation.assert_called_once()
        mock_services["storage"].save_processing_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_with_alignment_comparison(self, pipeline, mock_services):
        """Test pipeline with A/B testing of alignment strategies."""
        test_url = "https://youtube.com/watch?v=test"
        strategies = [AlignmentStrategy.LENGTH_BASED, AlignmentStrategy.HYBRID]
        
        # Execute A/B testing
        results = await pipeline.process_video_with_alignment_comparison(test_url, strategies)
        
        # Verify results
        assert len(results) == 2
        assert "length_based" in results
        assert "hybrid" in results
        
        for strategy_name, result in results.items():
            assert result.status == ProcessingStatus.COMPLETED
            assert result.video_id.endswith(f"_{strategy_name}")
            assert "alignment_evaluation" in result.files
            assert "comparison_summary" in result.files
        
        # Verify service calls
        mock_services["data_extraction"].extract_from_url.assert_called_once_with(test_url)
        mock_services["translation"].translate_batch.assert_called_once()
        assert mock_services["alignment"].align_translation.call_count == 2
        mock_services["alignment"].compare_alignments.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_data_extraction_failure(self, pipeline, mock_services):
        """Test pipeline behavior when data extraction fails."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Mock data extraction failure
        mock_services["data_extraction"].extract_from_url.side_effect = DataExtractionError("Failed to extract")
        
        # Execute pipeline
        result = await pipeline.process_video(test_url)
        
        # Verify failure handling
        assert result.status == ProcessingStatus.FAILED
        assert any("Failed to extract" in error for error in result.errors)
        
        # Verify subsequent services weren't called
        mock_services["translation"].translate_batch.assert_not_called()
        mock_services["alignment"].align_translation.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_translation_failure(self, pipeline, mock_services):
        """Test pipeline behavior when translation fails."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Mock translation failure
        mock_services["translation"].translate_batch.side_effect = TranslationError("Translation failed")
        
        # Execute pipeline
        result = await pipeline.process_video(test_url)
        
        # Verify failure handling
        assert result.status == ProcessingStatus.FAILED
        assert any("Translation failed" in error for error in result.errors)
        
        # Verify data extraction was called but alignment wasn't
        mock_services["data_extraction"].extract_from_url.assert_called_once()
        mock_services["alignment"].align_translation.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_alignment_failure(self, pipeline, mock_services):
        """Test pipeline behavior when alignment fails."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Mock alignment failure
        mock_services["alignment"].align_translation.side_effect = AlignmentError("Alignment failed")
        
        # Execute pipeline
        result = await pipeline.process_video(test_url)
        
        # Verify failure handling
        assert result.status == ProcessingStatus.FAILED
        assert any("Alignment failed" in error for error in result.errors)
        
        # Verify data extraction and translation were called
        mock_services["data_extraction"].extract_from_url.assert_called_once()
        mock_services["translation"].translate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_empty_transcript(self, pipeline, mock_services, sample_video_metadata):
        """Test pipeline behavior with empty transcript."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Mock empty transcript
        empty_transcript = TimedTranscript(
            segments=[TranscriptSegment(start_time=0.0, end_time=0.1, text="dummy")],  # Need at least one segment to avoid validation error
            source_type=SourceType.YOUTUBE,
            timing_metadata=TimingMetadata(
                segment_count=1,
                total_duration=0.1,
                average_segment_duration=0.1,
                timing_accuracy=0.0,
            ),
            video_metadata=sample_video_metadata,
            extraction_quality=0.0,
        )
        # Clear segments after creation to simulate empty
        empty_transcript.segments = []
        mock_services["data_extraction"].extract_from_url.return_value = empty_transcript
        
        # Execute pipeline
        result = await pipeline.process_video(test_url)
        
        # Verify failure handling
        assert result.status == ProcessingStatus.FAILED
        assert any("No transcript segments available" in error for error in result.errors)
        
        # Verify subsequent services weren't called
        mock_services["translation"].translate_batch.assert_not_called()
        mock_services["alignment"].align_translation.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_cost_tracking(self, pipeline, mock_services):
        """Test pipeline cost tracking integration."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Mock cost tracking
        with patch("videodub.core.pipeline.reset_global_cost_tracker") as mock_reset, \
             patch("videodub.core.pipeline.get_session_cost_summary") as mock_get_summary:
            
            mock_get_summary.return_value = {
                "total_cost": 0.25,
                "translation_cost": 0.15,
                "tts_cost": 0.10,
            }
            
            # Execute pipeline
            result = await pipeline.process_video(test_url)
            
            # Verify cost tracking
            mock_reset.assert_called_once()
            mock_get_summary.assert_called_once()
            assert result.cost_summary == {
                "total_cost": 0.25,
                "translation_cost": 0.15,
                "tts_cost": 0.10,
            }

    @pytest.mark.asyncio
    async def test_pipeline_service_integration_order(self, pipeline, mock_services):
        """Test that services are called in the correct order."""
        test_url = "https://youtube.com/watch?v=test"
        
        # Track call order
        call_order = []
        
        # Track call order by wrapping the original mock methods
        original_extract = mock_services["data_extraction"].extract_from_url
        original_translate = mock_services["translation"].translate_batch
        original_align = mock_services["alignment"].align_translation
        
        async def track_extract(*args, **kwargs):
            call_order.append("data_extraction")
            return await original_extract(*args, **kwargs)
        
        async def track_translate(*args, **kwargs):
            call_order.append("translation")
            return await original_translate(*args, **kwargs)
        
        async def track_align(*args, **kwargs):
            call_order.append("alignment")
            return await original_align(*args, **kwargs)
        
        mock_services["data_extraction"].extract_from_url = track_extract
        mock_services["translation"].translate_batch = track_translate
        mock_services["alignment"].align_translation = track_align
        
        # Execute pipeline
        result = await pipeline.process_video(test_url)
        
        # Verify call order
        expected_order = ["data_extraction", "translation", "alignment"]
        assert call_order == expected_order

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(self, pipeline, mock_services):
        """Test pipeline can handle concurrent processing requests."""
        test_urls = [
            "https://youtube.com/watch?v=test1",
            "https://youtube.com/watch?v=test2",
            "https://youtube.com/watch?v=test3",
        ]
        
        # Execute multiple pipelines concurrently
        tasks = [pipeline.process_video(url) for url in test_urls]
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        assert len(results) == 3
        for result in results:
            assert result.status == ProcessingStatus.COMPLETED
        
        # Verify services were called for each request
        assert mock_services["data_extraction"].extract_from_url.call_count == 3
        assert mock_services["translation"].translate_batch.call_count == 3
        assert mock_services["alignment"].align_translation.call_count == 3