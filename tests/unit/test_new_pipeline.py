"""Tests for the new pipeline with AlignmentService integration."""

import pytest
from unittest.mock import AsyncMock, Mock
from pathlib import Path

from videodub.core.exceptions import AlignmentError, DataExtractionError
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
from videodub.core.new_pipeline import NewTranslationPipeline


class TestNewTranslationPipeline:
    """Test NewTranslationPipeline implementation."""

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
            TranscriptSegment(start_time=80.0, end_time=120.0, text="Goodbye everyone."),
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
        return ["Hola mundo", "Esto es una prueba", "Adiós a todos."]

    @pytest.fixture
    def sample_timed_translation(self, sample_timed_transcript, sample_translated_texts):
        """Create sample timed translation."""
        segments = []
        for i, (orig_seg, trans_text) in enumerate(
            zip(sample_timed_transcript.segments, sample_translated_texts)
        ):
            segments.append(
                TimedTranslationSegment(
                    start_time=orig_seg.start_time,
                    end_time=orig_seg.end_time,
                    original_text=orig_seg.text,
                    translated_text=trans_text,
                    alignment_confidence=0.85,
                )
            )

        evaluation = AlignmentEvaluation(
            strategy=AlignmentStrategy.HYBRID,
            timing_accuracy=0.9,
            text_preservation=0.95,
            boundary_alignment=0.8,
            overall_score=0.88,
            execution_time=1.2,
            segment_count=3,
            average_confidence=0.85,
        )

        return TimedTranslation(
            segments=segments,
            original_transcript=sample_timed_transcript,
            target_language="es",
            alignment_config=AlignmentConfig(strategy=AlignmentStrategy.HYBRID),
            alignment_evaluation=evaluation,
            timing_metadata=sample_timed_transcript.timing_metadata,
        )

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return {
            "data_extraction": Mock(spec=DataExtractionService),
            "translation": Mock(spec=TranslationService),
            "alignment": Mock(spec=AlignmentService),
            "tts": Mock(spec=TTSService),
            "audio": Mock(spec=AudioProcessingService),
            "video": Mock(spec=VideoProcessingService),
            "storage": Mock(spec=StorageService),
        }

    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configuration."""
        return PipelineConfig(
            output_directory=Path("/tmp/test_output"),
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
        )

    @pytest.fixture
    def new_pipeline(self, mock_services, pipeline_config):
        """Create NewTranslationPipeline instance."""
        return NewTranslationPipeline(
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
    async def test_process_video_success(
        self,
        new_pipeline,
        mock_services,
        sample_timed_transcript,
        sample_translated_texts,
        sample_timed_translation,
    ):
        """Test successful video processing."""
        # Setup mocks
        mock_services["data_extraction"].extract_from_url = AsyncMock(
            return_value=sample_timed_transcript
        )
        mock_services["translation"].translate_batch = AsyncMock(
            return_value=sample_translated_texts
        )
        mock_services["alignment"].align_translation = AsyncMock(
            return_value=sample_timed_translation
        )
        async def mock_audio_generator(job):
            yield Path("/tmp/audio1.wav")
            yield Path("/tmp/audio2.wav")
            yield Path("/tmp/audio3.wav")
        
        mock_services["tts"].generate_batch_audio = mock_audio_generator
        mock_services["storage"].get_video_directory = AsyncMock(
            return_value=Path("/tmp/video_dir")
        )
        mock_services["storage"].save_metadata = AsyncMock(
            return_value=Path("/tmp/metadata.json")
        )
        mock_services["storage"].save_timed_transcript = AsyncMock(
            return_value=Path("/tmp/transcript.json")
        )
        mock_services["storage"].save_timed_translation = AsyncMock(
            return_value=Path("/tmp/translation.json")
        )
        mock_services["storage"].save_processing_result = AsyncMock(
            return_value=Path("/tmp/result.json")
        )
        mock_services["audio"].combine_audio_segments = AsyncMock()

        # Process video
        result = await new_pipeline.process_video("https://youtube.com/watch?v=test")

        # Verify result
        assert result.status == ProcessingStatus.COMPLETED
        assert result.target_language == "es"
        assert result.metadata == sample_timed_transcript.video_metadata

        # Verify service calls
        mock_services["data_extraction"].extract_from_url.assert_called_once()
        mock_services["translation"].translate_batch.assert_called_once()
        mock_services["alignment"].align_translation.assert_called_once()
        mock_services["storage"].save_processing_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_video_data_extraction_failure(
        self, new_pipeline, mock_services
    ):
        """Test video processing with data extraction failure."""
        # Setup mock to fail
        mock_services["data_extraction"].extract_from_url = AsyncMock(
            side_effect=DataExtractionError("Failed to extract data")
        )
        mock_services["storage"].save_processing_result = AsyncMock()

        # Process video
        result = await new_pipeline.process_video("https://youtube.com/watch?v=test")

        # Verify failure
        assert result.status == ProcessingStatus.FAILED
        assert "Failed to extract data" in result.errors[0]

    @pytest.mark.asyncio
    async def test_process_video_alignment_failure(
        self,
        new_pipeline,
        mock_services,
        sample_timed_transcript,
        sample_translated_texts,
    ):
        """Test video processing with alignment failure."""
        # Setup mocks
        mock_services["data_extraction"].extract_from_url = AsyncMock(
            return_value=sample_timed_transcript
        )
        mock_services["translation"].translate_batch = AsyncMock(
            return_value=sample_translated_texts
        )
        mock_services["alignment"].align_translation = AsyncMock(
            side_effect=AlignmentError("Alignment failed")
        )
        mock_services["storage"].save_processing_result = AsyncMock()

        # Process video
        result = await new_pipeline.process_video("https://youtube.com/watch?v=test")

        # Verify failure
        assert result.status == ProcessingStatus.FAILED
        assert "Alignment failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_process_video_with_alignment_comparison(
        self,
        new_pipeline,
        mock_services,
        sample_timed_transcript,
        sample_translated_texts,
        sample_timed_translation,
    ):
        """Test A/B testing with multiple alignment strategies."""
        strategies = [AlignmentStrategy.LENGTH_BASED, AlignmentStrategy.HYBRID]

        # Create different timed translations for each strategy
        length_based_translation = sample_timed_translation
        length_based_translation.alignment_config.strategy = AlignmentStrategy.LENGTH_BASED
        length_based_translation.alignment_evaluation.strategy = AlignmentStrategy.LENGTH_BASED

        hybrid_translation = sample_timed_translation
        hybrid_translation.alignment_config.strategy = AlignmentStrategy.HYBRID
        hybrid_translation.alignment_evaluation.strategy = AlignmentStrategy.HYBRID

        # Setup mocks
        mock_services["data_extraction"].extract_from_url = AsyncMock(
            return_value=sample_timed_transcript
        )
        mock_services["translation"].translate_batch = AsyncMock(
            return_value=sample_translated_texts
        )
        mock_services["alignment"].align_translation = AsyncMock(
            side_effect=[length_based_translation, hybrid_translation]
        )
        mock_services["alignment"].compare_alignments = AsyncMock(
            return_value=[
                length_based_translation.alignment_evaluation,
                hybrid_translation.alignment_evaluation,
            ]
        )
        mock_services["storage"].save_processing_result = AsyncMock()

        # Process video with A/B testing
        results = await new_pipeline.process_video_with_alignment_comparison(
            "https://youtube.com/watch?v=test", strategies
        )

        # Verify results
        assert len(results) == 2
        assert "length_based" in results
        assert "hybrid" in results
        assert results["length_based"].status == ProcessingStatus.COMPLETED
        assert results["hybrid"].status == ProcessingStatus.COMPLETED

        # Verify comparison summary exists
        for result in results.values():
            assert "comparison_summary" in result.files
            assert "best_strategy" in result.files["comparison_summary"]

    @pytest.mark.asyncio
    async def test_extract_data(
        self, new_pipeline, mock_services, sample_timed_transcript
    ):
        """Test data extraction step."""
        mock_services["data_extraction"].extract_from_url = AsyncMock(
            return_value=sample_timed_transcript
        )

        timed_transcript = await new_pipeline._extract_data(
            "https://youtube.com/watch?v=test"
        )

        assert timed_transcript == sample_timed_transcript
        mock_services["data_extraction"].extract_from_url.assert_called_once_with(
            "https://youtube.com/watch?v=test"
        )

    @pytest.mark.asyncio
    async def test_translate_texts(
        self, new_pipeline, mock_services, sample_timed_transcript, sample_translated_texts
    ):
        """Test text translation step."""
        mock_services["translation"].translate_batch = AsyncMock(
            return_value=sample_translated_texts
        )

        translated_texts = await new_pipeline._translate_texts(
            "test_video", sample_timed_transcript, "es"
        )

        assert translated_texts == sample_translated_texts
        mock_services["translation"].translate_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_align_translation(
        self,
        new_pipeline,
        mock_services,
        sample_timed_transcript,
        sample_translated_texts,
        sample_timed_translation,
    ):
        """Test alignment step."""
        mock_services["alignment"].align_translation = AsyncMock(
            return_value=sample_timed_translation
        )

        timed_translation = await new_pipeline._align_translation(
            "test_video", sample_timed_transcript, sample_translated_texts, "es"
        )

        assert timed_translation == sample_timed_translation
        mock_services["alignment"].align_translation.assert_called_once()

    def test_extract_video_id(self, new_pipeline):
        """Test video ID extraction."""
        # YouTube URL with v= parameter
        video_id = new_pipeline._extract_video_id(
            "https://youtube.com/watch?v=abc123&t=30s"
        )
        assert video_id == "abc123"

        # YouTube short URL
        video_id = new_pipeline._extract_video_id("https://youtu.be/xyz789")
        assert video_id == "xyz789"

        # Non-YouTube URL (fallback)
        video_id = new_pipeline._extract_video_id("https://example.com/video")
        assert video_id.startswith("video_")