"""Performance integration tests for the videodub pipeline."""

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from typing import List

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


class TestPerformanceIntegration:
    """Performance integration tests for the pipeline."""

    @pytest.fixture
    def performance_video_metadata(self):
        """Create video metadata for performance testing."""
        return VideoMetadata(
            video_id="performance_test",
            title="Performance Test Video",
            duration=300.0,  # 5 minutes
            url="https://youtube.com/watch?v=performance_test",
            channel="Performance Test Channel",
        )

    @pytest.fixture
    def large_timed_transcript(self, performance_video_metadata):
        """Create a large timed transcript for performance testing."""
        # Create 100 segments for performance testing
        segments = []
        for i in range(100):
            start_time = i * 3.0
            end_time = (i + 1) * 3.0
            segments.append(
                TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=f"This is segment {i + 1} with some realistic text content that would be found in a video transcript.",
                )
            )
        
        timing_metadata = TimingMetadata(
            segment_count=100,
            total_duration=300.0,
            average_segment_duration=3.0,
            timing_accuracy=0.95,
        )
        
        return TimedTranscript(
            segments=segments,
            source_type=SourceType.YOUTUBE,
            timing_metadata=timing_metadata,
            video_metadata=performance_video_metadata,
            extraction_quality=0.9,
        )

    @pytest.fixture
    def performance_mock_services(self, tmp_path, large_timed_transcript):
        """Create mock services with realistic performance characteristics."""
        # Mock DataExtractionService with realistic delay
        data_extraction_service = Mock(spec=DataExtractionService)
        
        async def mock_extract_from_url(url):
            await asyncio.sleep(0.1)  # Simulate 100ms extraction time
            return large_timed_transcript
        
        data_extraction_service.extract_from_url = mock_extract_from_url
        
        # Mock TranslationService with realistic delay
        translation_service = Mock(spec=TranslationService)
        
        async def mock_translate_batch(texts, target_language):
            await asyncio.sleep(0.05 * len(texts))  # 50ms per text
            return [f"Translated: {text}" for text in texts]
        
        translation_service.translate_batch = mock_translate_batch
        
        # Mock AlignmentService with realistic delay
        alignment_service = Mock(spec=AlignmentService)
        
        async def mock_align_translation(timed_transcript, translated_texts, target_language, config):
            await asyncio.sleep(0.02 * len(translated_texts))  # 20ms per segment
            
            segments = []
            for i, (original_seg, translated_text) in enumerate(zip(timed_transcript.segments, translated_texts)):
                segments.append(
                    TimedTranslationSegment(
                        start_time=original_seg.start_time,
                        end_time=original_seg.end_time,
                        original_text=original_seg.text,
                        translated_text=translated_text,
                        alignment_confidence=0.9 - (i * 0.001),  # Slight confidence decrease
                    )
                )
            
            evaluation = AlignmentEvaluation(
                strategy=config.strategy,
                timing_accuracy=0.95,
                text_preservation=0.92,
                boundary_alignment=0.88,
                overall_score=0.92,
                execution_time=0.02 * len(translated_texts),
                segment_count=len(translated_texts),
                average_confidence=0.9,
            )
            
            timing_metadata = TimingMetadata(
                segment_count=len(segments),
                total_duration=max(seg.end_time for seg in segments) if segments else 0.0,
                average_segment_duration=3.0,
                timing_accuracy=0.95,
            )
            
            alignment_config = AlignmentConfig(strategy=config.strategy)
            
            return TimedTranslation(
                segments=segments,
                original_transcript=timed_transcript,
                target_language=target_language,
                alignment_config=alignment_config,
                alignment_evaluation=evaluation,
                timing_metadata=timing_metadata,
            )
        
        alignment_service.align_translation = mock_align_translation
        
        # Mock compare_alignments method
        async def mock_compare_alignments(timed_translations):
            return [translation.alignment_evaluation for translation in timed_translations]
        
        alignment_service.compare_alignments = mock_compare_alignments
        
        # Mock TTSService with realistic delay
        tts_service = Mock(spec=TTSService)
        
        async def mock_generate_batch_audio(segments, output_directory, language, voice=None):
            for i, segment in enumerate(segments):
                await asyncio.sleep(0.1)  # 100ms per audio file
                audio_path = tmp_path / f"audio_{i}.wav"
                audio_path.write_text("fake audio data")
                yield audio_path
        
        tts_service.generate_batch_audio = mock_generate_batch_audio
        
        # Mock other services with minimal delays
        audio_service = Mock(spec=AudioProcessingService)
        audio_service.combine_audio_segments = AsyncMock(return_value=tmp_path / "combined.wav")
        
        video_service = Mock(spec=VideoProcessingService)
        video_service.create_dubbed_video = AsyncMock(return_value=tmp_path / "dubbed.mp4")
        
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
    def performance_pipeline(self, performance_mock_services, tmp_path):
        """Create pipeline for performance testing."""
        config = PipelineConfig(
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
            output_directory=tmp_path,
        )
        
        return NewTranslationPipeline(
            data_extraction_service=performance_mock_services["data_extraction"],
            translation_service=performance_mock_services["translation"],
            alignment_service=performance_mock_services["alignment"],
            tts_service=performance_mock_services["tts"],
            audio_service=performance_mock_services["audio"],
            video_processing_service=performance_mock_services["video"],
            storage_service=performance_mock_services["storage"],
            config=config,
        )

    @pytest.mark.asyncio
    async def test_pipeline_processing_time(self, performance_pipeline):
        """Test pipeline processing time for large transcript."""
        test_url = "https://youtube.com/watch?v=performance_test"
        
        start_time = time.time()
        result = await performance_pipeline.process_video(test_url)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify result
        assert result.status.value == "completed"
        
        # Performance assertions
        # Expected times: 0.1s (data) + 5s (translation) + 2s (alignment) + 10s (tts) = ~17s
        assert processing_time < 20.0, f"Processing took {processing_time:.2f}s, expected < 20s"
        
        # Log performance metrics
        print(f"Pipeline processing time: {processing_time:.2f}s for 100 segments")

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_processing(self, performance_pipeline):
        """Test concurrent processing of multiple videos."""
        test_urls = [
            "https://youtube.com/watch?v=test1",
            "https://youtube.com/watch?v=test2",
            "https://youtube.com/watch?v=test3",
        ]
        
        start_time = time.time()
        
        # Process videos concurrently
        tasks = [performance_pipeline.process_video(url) for url in test_urls]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify all results
        assert len(results) == 3
        for result in results:
            assert result.status.value == "completed"
        
        # Concurrent processing should be significantly faster than sequential
        # Sequential would be ~60s, concurrent should be ~20s
        assert concurrent_time < 25.0, f"Concurrent processing took {concurrent_time:.2f}s, expected < 25s"
        
        print(f"Concurrent processing time: {concurrent_time:.2f}s for 3 videos")

    @pytest.mark.asyncio
    async def test_alignment_strategy_performance_comparison(self, performance_pipeline, large_timed_transcript):
        """Test performance comparison of different alignment strategies."""
        test_url = "https://youtube.com/watch?v=performance_test"
        strategies = [
            AlignmentStrategy.LENGTH_BASED,
            AlignmentStrategy.SENTENCE_BOUNDARY,
            AlignmentStrategy.HYBRID,
        ]
        
        start_time = time.time()
        
        # Test A/B comparison with multiple strategies
        results = await performance_pipeline.process_video_with_alignment_comparison(test_url, strategies)
        
        end_time = time.time()
        comparison_time = end_time - start_time
        
        # Verify results
        assert len(results) == 3
        for strategy_name, result in results.items():
            assert result.status.value == "completed"
            assert "alignment_evaluation" in result.files
            assert "comparison_summary" in result.files
        
        # A/B testing should be reasonably fast
        # Expected: data extraction (0.1s) + translation (5s) + 3 alignments (6s) = ~11s
        assert comparison_time < 15.0, f"A/B comparison took {comparison_time:.2f}s, expected < 15s"
        
        print(f"A/B testing time: {comparison_time:.2f}s for 3 strategies")

    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, performance_pipeline):
        """Test memory usage with different transcript sizes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a large video
        test_url = "https://youtube.com/watch?v=performance_test"
        result = await performance_pipeline.process_video(test_url)
        
        # Get memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify result
        assert result.status.value == "completed"
        
        # Memory usage should be reasonable
        assert memory_increase < 500, f"Memory increased by {memory_increase:.2f}MB, expected < 500MB"
        
        print(f"Memory usage increased by {memory_increase:.2f}MB")

    @pytest.mark.asyncio
    async def test_processing_throughput(self, performance_pipeline):
        """Test processing throughput (segments per second)."""
        test_url = "https://youtube.com/watch?v=performance_test"
        
        start_time = time.time()
        result = await performance_pipeline.process_video(test_url)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify result and calculate throughput
        assert result.status.value == "completed"
        
        # Calculate throughput (segments per second)
        segments_processed = 100  # From large_timed_transcript
        throughput = segments_processed / processing_time
        
        # Throughput should be reasonable
        assert throughput > 3.0, f"Throughput was {throughput:.2f} segments/s, expected > 3.0"
        
        print(f"Processing throughput: {throughput:.2f} segments/second")

    @pytest.mark.asyncio
    async def test_service_bottleneck_identification(self, performance_mock_services, tmp_path):
        """Test identification of service bottlenecks."""
        config = PipelineConfig(
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
            output_directory=tmp_path,
        )
        
        # Create pipeline with timing tracking
        pipeline = NewTranslationPipeline(
            data_extraction_service=performance_mock_services["data_extraction"],
            translation_service=performance_mock_services["translation"],
            alignment_service=performance_mock_services["alignment"],
            tts_service=performance_mock_services["tts"],
            audio_service=performance_mock_services["audio"],
            video_processing_service=performance_mock_services["video"],
            storage_service=performance_mock_services["storage"],
            config=config,
        )
        
        # Track service execution times
        service_times = {}
        
        # Wrap services with timing
        original_extract = performance_mock_services["data_extraction"].extract_from_url
        original_translate = performance_mock_services["translation"].translate_batch
        original_align = performance_mock_services["alignment"].align_translation
        
        async def timed_extract(url):
            start = time.time()
            result = await original_extract(url)
            service_times["data_extraction"] = time.time() - start
            return result
        
        async def timed_translate(texts, target_language):
            start = time.time()
            result = await original_translate(texts, target_language)
            service_times["translation"] = time.time() - start
            return result
        
        async def timed_align(timed_transcript, translated_texts, target_language, config):
            start = time.time()
            result = await original_align(timed_transcript, translated_texts, target_language, config)
            service_times["alignment"] = time.time() - start
            return result
        
        performance_mock_services["data_extraction"].extract_from_url = timed_extract
        performance_mock_services["translation"].translate_batch = timed_translate
        performance_mock_services["alignment"].align_translation = timed_align
        
        # Process video
        test_url = "https://youtube.com/watch?v=performance_test"
        result = await pipeline.process_video(test_url)
        
        # Verify result
        assert result.status.value == "completed"
        
        # Analyze bottlenecks
        assert "data_extraction" in service_times
        assert "translation" in service_times
        assert "alignment" in service_times
        
        # Find bottleneck
        bottleneck_service = max(service_times.keys(), key=lambda k: service_times[k])
        bottleneck_time = service_times[bottleneck_service]
        
        print(f"Service execution times: {service_times}")
        print(f"Bottleneck: {bottleneck_service} ({bottleneck_time:.2f}s)")
        
        # Translation should be the bottleneck for 100 segments
        assert bottleneck_service == "translation", f"Expected translation bottleneck, got {bottleneck_service}"

    @pytest.mark.asyncio
    async def test_error_handling_performance(self, performance_pipeline, performance_mock_services):
        """Test performance impact of error handling."""
        test_url = "https://youtube.com/watch?v=performance_test"
        
        # Test with translation error
        performance_mock_services["translation"].translate_batch = AsyncMock(
            side_effect=Exception("Translation error")
        )
        
        start_time = time.time()
        result = await performance_pipeline.process_video(test_url)
        end_time = time.time()
        
        error_handling_time = end_time - start_time
        
        # Verify error handling
        assert result.status.value == "failed"
        assert "Translation error" in result.error_message
        
        # Error handling should be fast
        assert error_handling_time < 1.0, f"Error handling took {error_handling_time:.2f}s, expected < 1s"
        
        print(f"Error handling time: {error_handling_time:.2f}s")

    @pytest.mark.asyncio
    async def test_resource_cleanup_performance(self, performance_pipeline):
        """Test resource cleanup performance."""
        test_url = "https://youtube.com/watch?v=performance_test"
        
        # Process video multiple times to test resource cleanup
        for i in range(3):
            start_time = time.time()
            result = await performance_pipeline.process_video(test_url)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify result
            assert result.status.value == "completed"
            
            # Processing time should be consistent (no resource leaks)
            assert processing_time < 25.0, f"Processing iteration {i+1} took {processing_time:.2f}s"
            
            print(f"Processing iteration {i+1}: {processing_time:.2f}s")

    @pytest.mark.asyncio
    async def test_scalability_limits(self, performance_mock_services, tmp_path):
        """Test pipeline scalability limits."""
        # Test with extremely large transcript (1000 segments)
        large_segments = []
        for i in range(1000):
            start_time = i * 0.5
            end_time = (i + 1) * 0.5
            large_segments.append(
                TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=f"Segment {i + 1}",
                )
            )
        
        large_transcript = TimedTranscript(
            segments=large_segments,
            video_metadata=VideoMetadata(
                video_id="scalability_test",
                title="Scalability Test",
                duration=500.0,
                url="https://youtube.com/watch?v=scalability_test",
                channel="Test Channel",
            ),
            timing_metadata=TimingMetadata(
                segment_count=1000,
                total_duration=500.0,
                average_segment_duration=0.5,
                timing_accuracy=0.95,
            ),
            source_type=SourceType.YOUTUBE,
            extraction_quality=0.9,
            created_at=None,
        )
        
        # Update mock to return large transcript
        performance_mock_services["data_extraction"].extract_from_url = AsyncMock(
            return_value=large_transcript
        )
        
        config = PipelineConfig(
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
            output_directory=tmp_path,
        )
        
        pipeline = NewTranslationPipeline(
            data_extraction_service=performance_mock_services["data_extraction"],
            translation_service=performance_mock_services["translation"],
            alignment_service=performance_mock_services["alignment"],
            tts_service=performance_mock_services["tts"],
            audio_service=performance_mock_services["audio"],
            video_processing_service=performance_mock_services["video"],
            storage_service=performance_mock_services["storage"],
            config=config,
        )
        
        test_url = "https://youtube.com/watch?v=scalability_test"
        
        start_time = time.time()
        result = await pipeline.process_video(test_url)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify result
        assert result.status.value == "completed"
        
        # Should handle 1000 segments within reasonable time
        assert processing_time < 120.0, f"Large scale processing took {processing_time:.2f}s, expected < 120s"
        
        print(f"1000 segment processing time: {processing_time:.2f}s")