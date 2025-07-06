"""New translation pipeline with AlignmentService integration."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from ..config.validation import validate_language_code, validate_video_url
from ..utils.cost_tracking import get_session_cost_summary, reset_global_cost_tracker
from .exceptions import (
    AlignmentError,
    DataExtractionError,
    PipelineError,
    TranslationError,
    TTSError,
)
from .interfaces import (
    AlignmentService,
    AudioProcessingService,
    DataExtractionService,
    StorageService,
    TranslationService,
    TTSService,
    VideoProcessingService,
)
from .models import (
    AlignmentConfig,
    AlignmentStrategy,
    PipelineConfig,
    ProcessingResult,
    ProcessingStatus,
    TimedTranslation,
    TimedTranscript,
    TranslationSegment,
    TTSEngine,
)

logger = structlog.get_logger(__name__)


class NewTranslationPipeline:
    """New translation pipeline with AlignmentService integration."""

    def __init__(
        self,
        data_extraction_service: DataExtractionService,
        translation_service: TranslationService,
        alignment_service: AlignmentService,
        tts_service: TTSService,
        audio_service: AudioProcessingService,
        video_processing_service: VideoProcessingService,
        storage_service: StorageService,
        config: PipelineConfig,
        alignment_config: Optional[AlignmentConfig] = None,
    ):
        """
        Initialize the new translation pipeline.

        Args:
            data_extraction_service: Data extraction service
            translation_service: Translation service
            alignment_service: Alignment service
            tts_service: Text-to-speech service
            audio_service: Audio processing service
            video_processing_service: Video processing service
            storage_service: Storage service
            config: Pipeline configuration
            alignment_config: Alignment configuration (optional)
        """
        self.data_extraction_service = data_extraction_service
        self.translation_service = translation_service
        self.alignment_service = alignment_service
        self.tts_service = tts_service
        self.audio_service = audio_service
        self.video_processing_service = video_processing_service
        self.storage_service = storage_service
        self.config = config
        self.alignment_config = alignment_config or AlignmentConfig(
            strategy=AlignmentStrategy.HYBRID
        )

        logger.info(
            "New translation pipeline initialized",
            target_language=config.target_language,
            tts_engine=config.tts_engine.value,
            alignment_strategy=self.alignment_config.strategy.value,
            output_directory=str(config.output_directory),
        )

    async def process_video(
        self, url: str, video_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single video through the complete pipeline.

        Args:
            url: Video URL to process
            video_id: Optional custom video ID

        Returns:
            Processing result with status and file paths
        """
        validate_video_url(url)
        validate_language_code(self.config.target_language)

        # Extract video ID if not provided
        if not video_id:
            video_id = self._extract_video_id(url)

        # Initialize processing result
        result = ProcessingResult(
            video_id=video_id,
            status=ProcessingStatus.PROCESSING,
            target_language=self.config.target_language,
            tts_engine=self.config.tts_engine,
        )

        logger.info("Starting video processing", video_id=video_id, url=url)

        # Reset cost tracking for this processing session
        reset_global_cost_tracker()

        try:
            # Step 1: Extract data (transcript + timing)
            timed_transcript = await self._extract_data(url)
            result.metadata = timed_transcript.video_metadata

            if not timed_transcript.segments:
                result.mark_failed("No transcript segments available for translation")
                await self.storage_service.save_processing_result(result)
                return result

            # Step 2: Translate text (pure text-to-text)
            translated_texts = await self._translate_texts(
                video_id, timed_transcript, self.config.target_language
            )

            # Step 3: Align translation with timing
            timed_translation = await self._align_translation(
                video_id, timed_transcript, translated_texts, self.config.target_language
            )

            # Step 4: Generate audio for aligned translation
            audio_files = await self._generate_audio(
                video_id, timed_translation, self.config.target_language
            )

            # Step 5: Create dubbed video (optional)
            dubbed_video_path = await self._create_dubbed_video(
                video_id, timed_translation
            )

            # Step 6: Save all data to storage
            await self._save_pipeline_data(
                video_id, timed_transcript, timed_translation, result
            )

            # Step 7: Capture cost tracking data
            result.cost_summary = get_session_cost_summary()

            # Mark as completed
            result.mark_completed()

            logger.info(
                "Video processing completed",
                video_id=video_id,
                original_segments=len(timed_transcript.segments),
                aligned_segments=len(timed_translation.segments),
                audio_files=len(audio_files),
                alignment_strategy=timed_translation.alignment_strategy.value,
                alignment_quality=timed_translation.overall_quality,
                dubbed_video=str(dubbed_video_path) if dubbed_video_path else "None",
            )

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error("Video processing failed", video_id=video_id, error=error_msg)
            result.mark_failed(error_msg)

        # Always save the result
        await self.storage_service.save_processing_result(result)
        return result

    async def process_video_with_alignment_comparison(
        self, url: str, strategies: List[AlignmentStrategy], video_id: Optional[str] = None
    ) -> Dict[str, ProcessingResult]:
        """
        Process video with multiple alignment strategies for A/B testing.

        Args:
            url: Video URL to process
            strategies: List of alignment strategies to test
            video_id: Optional custom video ID

        Returns:
            Dict mapping strategy name to processing result
        """
        validate_video_url(url)
        validate_language_code(self.config.target_language)

        if not video_id:
            video_id = self._extract_video_id(url)

        logger.info(
            "Starting A/B testing with multiple alignment strategies",
            video_id=video_id,
            strategies=[s.value for s in strategies],
        )

        # Reset cost tracking
        reset_global_cost_tracker()

        results = {}

        try:
            # Step 1: Extract data once (shared across all strategies)
            timed_transcript = await self._extract_data(url)

            if not timed_transcript.segments:
                error_msg = "No transcript segments available for translation"
                for strategy in strategies:
                    result = ProcessingResult(
                        video_id=f"{video_id}_{strategy.value}",
                        status=ProcessingStatus.FAILED,
                        target_language=self.config.target_language,
                        tts_engine=self.config.tts_engine,
                    )
                    result.mark_failed(error_msg)
                    results[strategy.value] = result
                return results

            # Step 2: Translate text once (shared across all strategies)
            translated_texts = await self._translate_texts(
                video_id, timed_transcript, self.config.target_language
            )

            # Step 3: Test each alignment strategy
            timed_translations = []
            for strategy in strategies:
                try:
                    strategy_config = AlignmentConfig(
                        strategy=strategy,
                        parameters=self.alignment_config.parameters.copy(),
                    )
                    
                    timed_translation = await self.alignment_service.align_translation(
                        timed_transcript=timed_transcript,
                        translated_texts=translated_texts,
                        target_language=self.config.target_language,
                        config=strategy_config,
                    )
                    timed_translations.append(timed_translation)

                    # Create result for this strategy
                    strategy_video_id = f"{video_id}_{strategy.value}"
                    result = ProcessingResult(
                        video_id=strategy_video_id,
                        status=ProcessingStatus.COMPLETED,
                        target_language=self.config.target_language,
                        tts_engine=self.config.tts_engine,
                        metadata=timed_transcript.video_metadata,
                    )

                    # Add alignment metrics to result
                    result.files["alignment_evaluation"] = {
                        "strategy": strategy.value,
                        "timing_accuracy": timed_translation.alignment_evaluation.timing_accuracy,
                        "text_preservation": timed_translation.alignment_evaluation.text_preservation,
                        "boundary_alignment": timed_translation.alignment_evaluation.boundary_alignment,
                        "overall_score": timed_translation.alignment_evaluation.overall_score,
                        "execution_time": timed_translation.alignment_evaluation.execution_time,
                        "average_confidence": timed_translation.average_confidence,
                    }

                    result.mark_completed()
                    results[strategy.value] = result

                    logger.info(
                        "Strategy completed",
                        strategy=strategy.value,
                        quality=timed_translation.overall_quality,
                        execution_time=timed_translation.alignment_evaluation.execution_time,
                    )

                except Exception as e:
                    logger.error(
                        "Strategy failed",
                        strategy=strategy.value,
                        error=str(e),
                    )
                    result = ProcessingResult(
                        video_id=f"{video_id}_{strategy.value}",
                        status=ProcessingStatus.FAILED,
                        target_language=self.config.target_language,
                        tts_engine=self.config.tts_engine,
                    )
                    result.mark_failed(f"Alignment strategy failed: {str(e)}")
                    results[strategy.value] = result

            # Step 4: Compare alignment results
            if timed_translations:
                evaluations = await self.alignment_service.compare_alignments(
                    timed_translations
                )
                
                # Add comparison summary
                comparison_summary = {
                    "strategies_tested": len(strategies),
                    "successful_alignments": len(timed_translations),
                    "best_strategy": max(
                        evaluations, key=lambda e: e.overall_score
                    ).strategy.value,
                    "quality_scores": {
                        eval.strategy.value: eval.overall_score for eval in evaluations
                    },
                }

                # Add comparison summary to all results
                for result in results.values():
                    result.files["comparison_summary"] = comparison_summary

            logger.info(
                "A/B testing completed",
                video_id=video_id,
                strategies_tested=len(strategies),
                successful_alignments=len(timed_translations),
            )

        except Exception as e:
            error_msg = f"A/B testing pipeline failed: {str(e)}"
            logger.error("A/B testing failed", video_id=video_id, error=error_msg)
            
            # Mark all results as failed
            for strategy in strategies:
                if strategy.value not in results:
                    result = ProcessingResult(
                        video_id=f"{video_id}_{strategy.value}",
                        status=ProcessingStatus.FAILED,
                        target_language=self.config.target_language,
                        tts_engine=self.config.tts_engine,
                    )
                    result.mark_failed(error_msg)
                    results[strategy.value] = result

        # Save all results
        for result in results.values():
            await self.storage_service.save_processing_result(result)

        return results

    async def _extract_data(self, url: str) -> TimedTranscript:
        """Extract transcript and timing data from video."""
        try:
            logger.debug("Extracting data from URL", url=url)
            
            timed_transcript = await self.data_extraction_service.extract_from_url(url)

            logger.info(
                "Data extraction completed",
                video_id=timed_transcript.video_metadata.video_id,
                segments=len(timed_transcript.segments),
                duration=timed_transcript.timing_metadata.total_duration,
                quality=timed_transcript.extraction_quality,
                source_type=timed_transcript.source_type.value,
            )

            return timed_transcript

        except Exception as e:
            logger.error("Data extraction failed", url=url, error=str(e))
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"Failed to extract data: {str(e)}")

    async def _translate_texts(
        self, video_id: str, timed_transcript: TimedTranscript, target_language: str
    ) -> List[str]:
        """Translate transcript texts using pure text-to-text translation."""
        try:
            logger.debug(
                "Starting text translation",
                video_id=video_id,
                segment_count=len(timed_transcript.segments),
                target_language=target_language,
            )

            # Extract texts from segments
            original_texts = [segment.text for segment in timed_transcript.segments]

            # Translate using batch translation
            translated_texts = await self.translation_service.translate_batch(
                texts=original_texts,
                target_language=target_language,
            )

            logger.info(
                "Text translation completed",
                video_id=video_id,
                original_texts=len(original_texts),
                translated_texts=len(translated_texts),
            )

            return translated_texts

        except Exception as e:
            logger.error("Text translation failed", video_id=video_id, error=str(e))
            if isinstance(e, TranslationError):
                raise
            raise TranslationError(f"Failed to translate texts: {str(e)}")

    async def _align_translation(
        self,
        video_id: str,
        timed_transcript: TimedTranscript,
        translated_texts: List[str],
        target_language: str,
    ) -> TimedTranslation:
        """Align translated texts with original timing."""
        try:
            logger.debug(
                "Starting timing alignment",
                video_id=video_id,
                strategy=self.alignment_config.strategy.value,
            )

            timed_translation = await self.alignment_service.align_translation(
                timed_transcript=timed_transcript,
                translated_texts=translated_texts,
                target_language=target_language,
                config=self.alignment_config,
            )

            logger.info(
                "Timing alignment completed",
                video_id=video_id,
                strategy=timed_translation.alignment_strategy.value,
                segments=len(timed_translation.segments),
                quality=timed_translation.overall_quality,
                average_confidence=timed_translation.average_confidence,
            )

            return timed_translation

        except Exception as e:
            logger.error("Timing alignment failed", video_id=video_id, error=str(e))
            if isinstance(e, AlignmentError):
                raise
            raise AlignmentError(f"Failed to align translation: {str(e)}")

    async def _generate_audio(
        self, video_id: str, timed_translation: TimedTranslation, language: str
    ) -> List[Path]:
        """Generate audio for aligned translation segments."""
        try:
            logger.debug(
                "Starting audio generation",
                video_id=video_id,
                segment_count=len(timed_translation.segments),
                language=language,
            )

            # Create audio generation job from timed translation
            video_dir = await self.storage_service.get_video_directory(video_id)
            audio_dir = video_dir / "translated_audio"

            # Convert timed translation segments to translation segments for TTS
            from .models import TranslationSegment, TranscriptSegment
            
            translation_segments = []
            for timed_seg in timed_translation.segments:
                original_segment = TranscriptSegment(
                    start_time=timed_seg.start_time,
                    end_time=timed_seg.end_time,
                    text=timed_seg.original_text,
                )
                
                translation_segment = TranslationSegment(
                    original_segment=original_segment,
                    translated_text=timed_seg.translated_text,
                    language=language,
                )
                translation_segments.append(translation_segment)

            # Generate audio files
            audio_files = []
            async for audio_path in self.tts_service.generate_batch_audio(
                segments=translation_segments,
                output_directory=audio_dir,
                language=language,
            ):
                audio_files.append(audio_path)

                # Update segment with audio path
                segment_index = len(audio_files) - 1
                if segment_index < len(translation_segments):
                    translation_segments[segment_index].audio_path = audio_path

            logger.info(
                "Audio generation completed",
                video_id=video_id,
                audio_files=len(audio_files),
            )

            # Combine audio segments into single file
            if audio_files:
                combined_audio_path = audio_dir / "translated_audio.wav"
                await self.audio_service.combine_audio_segments(
                    audio_files=audio_files,
                    output_path=combined_audio_path,
                    segments=translation_segments,
                )

                logger.info(
                    "Audio combination completed",
                    video_id=video_id,
                    combined_file=combined_audio_path,
                )

            return audio_files

        except Exception as e:
            logger.error("Audio generation failed", video_id=video_id, error=str(e))
            if isinstance(e, TTSError):
                raise
            raise TTSError(f"Failed to generate audio: {str(e)}")

    async def _create_dubbed_video(
        self, video_id: str, timed_translation: TimedTranslation
    ) -> Optional[Path]:
        """Create dubbed video by combining original video with translated audio."""
        try:
            logger.info("Starting video dubbing", video_id=video_id)

            # Get video directories
            video_dir = await self.storage_service.get_video_directory(video_id)
            audio_dir = video_dir / "translated_audio"

            # Check if combined audio exists
            combined_audio_path = audio_dir / "translated_audio.wav"
            if not combined_audio_path.exists():
                logger.warning(f"Combined audio not found: {combined_audio_path}")
                return None

            # Find original video file
            original_video_path = await self._find_original_video(video_id)
            if not original_video_path:
                logger.warning("No original video file found")
                return None

            # Create output path for dubbed video
            dubbed_video_path = (
                video_dir / f"dubbed_video_{self.config.target_language}.mp4"
            )

            # Convert timed translation segments to translation segments for video processing
            from .models import TranslationSegment, TranscriptSegment
            
            translation_segments = []
            for timed_seg in timed_translation.segments:
                original_segment = TranscriptSegment(
                    start_time=timed_seg.start_time,
                    end_time=timed_seg.end_time,
                    text=timed_seg.original_text,
                )
                
                translation_segment = TranslationSegment(
                    original_segment=original_segment,
                    translated_text=timed_seg.translated_text,
                    language=self.config.target_language,
                )
                translation_segments.append(translation_segment)

            # Create dubbed video
            result_path = await self.video_processing_service.create_dubbed_video(
                original_video_path=original_video_path,
                translated_audio_path=combined_audio_path,
                output_path=dubbed_video_path,
                segments=translation_segments,
            )

            logger.info(
                "Video dubbing completed", video_id=video_id, output_path=result_path
            )

            return result_path

        except Exception as e:
            logger.error("Video dubbing failed", video_id=video_id, error=str(e))
            # Don't fail the entire pipeline for video dubbing errors
            logger.warning("Continuing without dubbed video due to error")
            return None

    async def _find_original_video(self, video_id: str) -> Optional[Path]:
        """Find original video file for dubbing."""
        try:
            # Look in the scraped directory
            scraped_dir = self.config.output_directory / "scraped" / video_id
            if scraped_dir.exists():
                video_files = (
                    list(scraped_dir.glob("*.mp4"))
                    + list(scraped_dir.glob("*.mkv"))
                    + list(scraped_dir.glob("*.webm"))
                )
                if video_files:
                    return video_files[0]

            # Look in video directory
            video_dir = await self.storage_service.get_video_directory(video_id)
            video_files = (
                list(video_dir.glob("*.mp4"))
                + list(video_dir.glob("*.mkv"))
                + list(video_dir.glob("*.webm"))
            )
            if video_files:
                return video_files[0]

            return None

        except Exception as e:
            logger.warning("Failed to find original video", video_id=video_id, error=str(e))
            return None

    async def _save_pipeline_data(
        self,
        video_id: str,
        timed_transcript: TimedTranscript,
        timed_translation: TimedTranslation,
        result: ProcessingResult,
    ) -> None:
        """Save all pipeline data to storage."""
        try:
            # Save video metadata
            metadata_path = await self.storage_service.save_metadata(
                video_id, timed_transcript.video_metadata
            )
            result.files["metadata"] = str(metadata_path)

            # Save timed transcript
            transcript_path = await self.storage_service.save_timed_transcript(
                video_id, timed_transcript
            )
            result.files["timed_transcript"] = str(transcript_path)

            # Save timed translation
            translation_path = await self.storage_service.save_timed_translation(
                video_id, timed_translation
            )
            result.files["timed_translation"] = str(translation_path)

            # Add audio directory to files
            video_dir = await self.storage_service.get_video_directory(video_id)
            audio_dir = video_dir / "translated_audio"
            if audio_dir.exists():
                result.files["audio_directory"] = str(audio_dir)

            # Add dubbed video to files if it exists
            dubbed_video_path = (
                video_dir / f"dubbed_video_{self.config.target_language}.mp4"
            )
            if dubbed_video_path.exists():
                result.files["dubbed_video"] = str(dubbed_video_path)

            # Add alignment evaluation
            result.files["alignment_evaluation"] = {
                "strategy": timed_translation.alignment_strategy.value,
                "timing_accuracy": timed_translation.alignment_evaluation.timing_accuracy,
                "text_preservation": timed_translation.alignment_evaluation.text_preservation,
                "boundary_alignment": timed_translation.alignment_evaluation.boundary_alignment,
                "overall_score": timed_translation.alignment_evaluation.overall_score,
                "execution_time": timed_translation.alignment_evaluation.execution_time,
                "average_confidence": timed_translation.average_confidence,
            }

            logger.debug("Pipeline data saved", video_id=video_id)

        except Exception as e:
            logger.error(
                "Failed to save pipeline data", video_id=video_id, error=str(e)
            )
            raise PipelineError(f"Failed to save pipeline data: {str(e)}")

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from URL for naming purposes."""
        try:
            if "youtube.com" in url or "youtu.be" in url:
                if "v=" in url:
                    return url.split("v=")[1].split("&")[0]
                elif "youtu.be/" in url:
                    return url.split("youtu.be/")[1].split("?")[0]

            # Fallback to hash-based ID
            return f"video_{hash(url) % 100000:05d}"

        except Exception:
            return f"video_{hash(url) % 100000:05d}"