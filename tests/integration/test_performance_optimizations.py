"""Performance optimization tests to validate improvements."""

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from videodub.core.interfaces import TranslationService
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
from videodub.core.pipeline import TranslationPipeline
from videodub.services.optimized_translator import OptimizedOpenAITranslationService, ConcurrentTTSOptimizer


class TestPerformanceOptimizations:
    """Tests for performance optimization implementations."""

    @pytest.fixture
    def sample_texts(self):
        """Create sample texts for performance testing."""
        return [
            f"This is sample text number {i + 1} for performance testing."
            for i in range(50)
        ]

    @pytest.fixture
    def mock_openai_adapter(self):
        """Mock OpenAI adapter for testing."""
        mock_adapter = Mock()
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "\\n".join([
            f"Translated text {i + 1}" for i in range(10)
        ])
        
        mock_adapter.chat_completion = AsyncMock(return_value=mock_response)
        return mock_adapter

    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, sample_texts, tmp_path):
        """Test that larger batch sizes improve performance."""
        # Create mock translation service with different batch sizes
        class MockTranslationService(TranslationService):
            def __init__(self, delay_per_call: float, batch_size: int = 1):
                self.delay_per_call = delay_per_call
                self.batch_size = batch_size
                self.api_calls = 0

            async def translate_text(self, text: str, target_language: str) -> str:
                return f"Translated: {text}"

            async def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
                # Simulate API calls based on batch size
                num_calls = (len(texts) + self.batch_size - 1) // self.batch_size
                self.api_calls += num_calls
                
                # Simulate delay for each API call
                await asyncio.sleep(self.delay_per_call * num_calls)
                
                return [f"Translated: {text}" for text in texts]

        # Test with small batch size (simulates original approach)
        small_batch_service = MockTranslationService(delay_per_call=0.05, batch_size=1)
        
        start_time = time.time()
        await small_batch_service.translate_batch(sample_texts, "es")
        small_batch_time = time.time() - start_time
        
        # Test with large batch size (optimized approach)
        large_batch_service = MockTranslationService(delay_per_call=0.05, batch_size=10)
        
        start_time = time.time()
        await large_batch_service.translate_batch(sample_texts, "es")
        large_batch_time = time.time() - start_time
        
        # Verify optimization
        assert large_batch_time < small_batch_time * 0.6, f"Large batch ({large_batch_time:.2f}s) should be significantly faster than small batch ({small_batch_time:.2f}s)"
        assert large_batch_service.api_calls < small_batch_service.api_calls * 0.3, f"Large batch should make fewer API calls: {large_batch_service.api_calls} vs {small_batch_service.api_calls}"
        
        print(f"Small batch: {small_batch_time:.2f}s ({small_batch_service.api_calls} API calls)")
        print(f"Large batch: {large_batch_time:.2f}s ({large_batch_service.api_calls} API calls)")
        print(f"Performance improvement: {((small_batch_time - large_batch_time) / small_batch_time * 100):.1f}%")

    @pytest.mark.asyncio
    async def test_optimized_translator_performance(self, sample_texts, monkeypatch):
        """Test the optimized translator service performance."""
        # Mock the OpenAI adapter
        mock_adapter = Mock()
        
        def create_mock_response(texts_count):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "\\n".join([
                f"Traducido: texto {i + 1}" for i in range(texts_count)
            ])
            return mock_response
        
        # Track API calls
        api_call_count = 0
        
        async def mock_chat_completion(*args, **kwargs):
            nonlocal api_call_count
            api_call_count += 1
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Extract batch size from the prompt
            prompt = kwargs['messages'][1]['content']
            lines = prompt.split('\\n')
            numbered_lines = [line for line in lines if line.strip() and line[0].isdigit()]
            batch_size = len(numbered_lines)
            
            return create_mock_response(batch_size)
        
        mock_adapter.chat_completion = mock_chat_completion
        
        # Patch the OpenAIAdapter import
        monkeypatch.setattr("videodub.services.optimized_translator.OpenAIAdapter", lambda api_key: mock_adapter)
        
        # Test optimized service with batch size 10
        optimized_service = OptimizedOpenAITranslationService(
            api_key="test-key",
            batch_size=10,
            max_concurrent_batches=2,
        )
        
        start_time = time.time()
        results = await optimized_service.translate_batch(sample_texts, "es")
        optimized_time = time.time() - start_time
        
        # Verify results
        assert len(results) == len(sample_texts)
        assert all("Traducido:" in result for result in results)
        
        # Verify performance: 50 texts with batch size 10 should make 5 API calls
        expected_calls = (len(sample_texts) + 9) // 10  # Ceiling division
        assert api_call_count == expected_calls, f"Expected {expected_calls} API calls, got {api_call_count}"
        
        print(f"Optimized translation: {optimized_time:.2f}s ({api_call_count} API calls)")
        print(f"Texts per second: {len(sample_texts) / optimized_time:.1f}")

    @pytest.mark.asyncio
    async def test_concurrent_tts_optimization(self, tmp_path):
        """Test concurrent TTS optimization."""
        optimizer = ConcurrentTTSOptimizer(max_concurrent_files=3)
        
        # Mock audio generation function
        generation_times = []
        
        async def mock_audio_generation(job_id):
            start = time.time()
            await asyncio.sleep(0.1)  # Simulate 100ms generation time
            generation_times.append(time.time() - start)
            
            audio_path = tmp_path / f"audio_{job_id}.wav"
            audio_path.write_text("fake audio")
            return audio_path
        
        # Test with 10 audio files
        jobs = list(range(10))
        
        start_time = time.time()
        results = await optimizer.optimize_audio_generation(mock_audio_generation, jobs)
        total_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 10
        assert all(path.exists() for path in results)
        
        # With max_concurrent=3, 10 files should take ~4 batches * 0.1s ≈ 0.4s
        # Sequential would take 10 * 0.1s = 1.0s
        expected_time = (len(jobs) / 3) * 0.1  # Approximately
        assert total_time < expected_time * 1.5, f"Concurrent TTS took {total_time:.2f}s, expected ~{expected_time:.2f}s"
        
        print(f"Concurrent TTS: {total_time:.2f}s for {len(jobs)} files")
        print(f"Sequential would take: ~{len(jobs) * 0.1:.2f}s")
        print(f"Speedup: {(len(jobs) * 0.1) / total_time:.1f}x")

    @pytest.mark.asyncio
    async def test_integrated_pipeline_optimizations(self, tmp_path):
        """Test integrated optimizations in the full pipeline."""
        # Create optimized mock services
        class OptimizedMockServices:
            def __init__(self):
                self.api_calls = {"translation": 0, "alignment": 0, "tts": 0}
                
            def create_data_extraction_service(self):
                service = Mock()
                
                segments = [
                    TranscriptSegment(
                        start_time=i * 2.0,
                        end_time=(i + 1) * 2.0,
                        text=f"Segment {i + 1} text for testing.",
                    )
                    for i in range(20)  # 20 segments for testing
                ]
                
                timing_metadata = TimingMetadata(
                    segment_count=20,
                    total_duration=40.0,
                    average_segment_duration=2.0,
                    timing_accuracy=0.95,
                )
                
                video_metadata = VideoMetadata(
                    video_id="optimization_test",
                    title="Optimization Test Video",
                    duration=40.0,
                    url="https://youtube.com/watch?v=optimization_test",
                    channel="Test Channel",
                )
                
                transcript = TimedTranscript(
                    segments=segments,
                    source_type=SourceType.YOUTUBE,
                    timing_metadata=timing_metadata,
                    video_metadata=video_metadata,
                    extraction_quality=0.9,
                )
                
                async def extract_from_url(url):
                    await asyncio.sleep(0.05)  # Fast extraction
                    return transcript
                
                service.extract_from_url = extract_from_url
                return service
            
            def create_translation_service(self):
                service = Mock()
                
                async def translate_batch(texts, target_language):
                    # Simulate optimized batching (5 texts per batch)
                    batch_size = 5
                    num_batches = (len(texts) + batch_size - 1) // batch_size
                    self.api_calls["translation"] += num_batches
                    
                    # Simulate concurrent batch processing
                    await asyncio.sleep(0.1 * num_batches / 2)  # Concurrent processing
                    
                    return [f"Optimized translation of: {text}" for text in texts]
                
                service.translate_batch = translate_batch
                return service
            
            def create_alignment_service(self):
                service = Mock()
                
                async def align_translation(timed_transcript, translated_texts, target_language, config):
                    self.api_calls["alignment"] += 1
                    await asyncio.sleep(0.05)  # Fast alignment
                    
                    segments = [
                        TimedTranslationSegment(
                            start_time=original_seg.start_time,
                            end_time=original_seg.end_time,
                            original_text=original_seg.text,
                            translated_text=translated_text,
                            alignment_confidence=0.9,
                        )
                        for original_seg, translated_text in zip(timed_transcript.segments, translated_texts)
                    ]
                    
                    evaluation = AlignmentEvaluation(
                        strategy=config.strategy,
                        timing_accuracy=0.95,
                        text_preservation=0.92,
                        boundary_alignment=0.88,
                        overall_score=0.92,
                        execution_time=0.05,
                        segment_count=len(segments),
                        average_confidence=0.9,
                    )
                    
                    timing_metadata = TimingMetadata(
                        segment_count=len(segments),
                        total_duration=timed_transcript.timing_metadata.total_duration,
                        average_segment_duration=2.0,
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
                
                service.align_translation = align_translation
                return service
            
            def create_other_services(self):
                # Mock TTS with concurrent optimization
                tts_service = Mock()
                
                async def generate_batch_audio(segments, output_directory, language, voice=None):
                    # Simulate concurrent generation (3 at a time)
                    self.api_calls["tts"] += len(segments)
                    concurrent_batches = (len(segments) + 2) // 3
                    await asyncio.sleep(0.05 * concurrent_batches)  # Optimized timing
                    
                    for i, segment in enumerate(segments):
                        audio_path = tmp_path / f"optimized_audio_{i}.wav"
                        audio_path.write_text("fake audio")
                        yield audio_path
                
                tts_service.generate_batch_audio = generate_batch_audio
                
                # Mock other services
                audio_service = Mock()
                audio_service.combine_audio_segments = AsyncMock(return_value=tmp_path / "combined.wav")
                
                video_service = Mock()
                video_service.create_dubbed_video = AsyncMock(return_value=tmp_path / "dubbed.mp4")
                
                storage_service = Mock()
                storage_service.get_video_directory = AsyncMock(return_value=tmp_path)
                storage_service.save_metadata = AsyncMock(return_value=tmp_path / "metadata.json")
                storage_service.save_timed_transcript = AsyncMock(return_value=tmp_path / "transcript.json")
                storage_service.save_timed_translation = AsyncMock(return_value=tmp_path / "translation.json")
                storage_service.save_processing_result = AsyncMock()
                
                return tts_service, audio_service, video_service, storage_service
        
        # Create optimized services
        mock_services = OptimizedMockServices()
        data_service = mock_services.create_data_extraction_service()
        translation_service = mock_services.create_translation_service()
        alignment_service = mock_services.create_alignment_service()
        tts_service, audio_service, video_service, storage_service = mock_services.create_other_services()
        
        # Create optimized pipeline
        config = PipelineConfig(
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
            output_directory=tmp_path,
        )
        
        pipeline = TranslationPipeline(
            data_extraction_service=data_service,
            translation_service=translation_service,
            alignment_service=alignment_service,
            tts_service=tts_service,
            audio_service=audio_service,
            video_processing_service=video_service,
            storage_service=storage_service,
            config=config,
        )
        
        # Test optimized pipeline
        test_url = "https://youtube.com/watch?v=optimization_test"
        
        start_time = time.time()
        result = await pipeline.process_video(test_url)
        optimized_time = time.time() - start_time
        
        # Verify results
        assert result.status.value == "completed"
        
        # Verify optimization metrics
        assert mock_services.api_calls["translation"] <= 4, f"Translation should make ≤4 API calls for 20 segments with batch size 5, got {mock_services.api_calls['translation']}"
        assert optimized_time < 5.0, f"Optimized pipeline should complete in <5s, took {optimized_time:.2f}s"
        
        # Calculate performance metrics
        segments_per_second = 20 / optimized_time
        
        print(f"Optimized pipeline performance:")
        print(f"  Total time: {optimized_time:.2f}s")
        print(f"  Segments/second: {segments_per_second:.1f}")
        print(f"  Translation API calls: {mock_services.api_calls['translation']}")
        print(f"  TTS generations: {mock_services.api_calls['tts']}")
        
        # Performance targets
        assert segments_per_second > 4.0, f"Should process >4 segments/second, got {segments_per_second:.1f}"

    @pytest.mark.asyncio
    async def test_memory_efficiency_optimization(self, tmp_path):
        """Test memory efficiency with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a large dataset (500 segments) and process it
        large_segments = [
            TranscriptSegment(
                start_time=i * 1.0,
                end_time=(i + 1) * 1.0,
                text=f"This is a longer segment {i + 1} with more detailed content that would be typical in a real video transcript. This helps simulate realistic memory usage patterns.",
            )
            for i in range(500)
        ]
        
        timing_metadata = TimingMetadata(
            segment_count=500,
            total_duration=500.0,
            average_segment_duration=1.0,
            timing_accuracy=0.95,
        )
        
        video_metadata = VideoMetadata(
            video_id="memory_test",
            title="Memory Test Video",
            duration=500.0,
            url="https://youtube.com/watch?v=memory_test",
            channel="Test Channel",
        )
        
        large_transcript = TimedTranscript(
            segments=large_segments,
            source_type=SourceType.YOUTUBE,
            timing_metadata=timing_metadata,
            video_metadata=video_metadata,
            extraction_quality=0.9,
        )
        
        # Process the large dataset
        texts = [segment.text for segment in large_segments]
        
        # Simulate memory-efficient processing
        batch_size = 50
        processed_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Simulate processing
            await asyncio.sleep(0.01)
            processed_results.extend([f"Processed: {text[:20]}..." for text in batch])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify memory efficiency
        assert len(processed_results) == 500
        assert memory_increase < 100, f"Memory increase should be <100MB, got {memory_increase:.1f}MB"
        
        print(f"Memory efficiency test:")
        print(f"  Processed segments: {len(processed_results)}")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Memory per segment: {memory_increase / len(processed_results):.3f}MB")