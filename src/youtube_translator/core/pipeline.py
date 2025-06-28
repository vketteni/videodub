"""Main translation pipeline implementation."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator

import structlog

from .interfaces import VideoScrapingService, TranslationService, TTSService, StorageService
from .models import (
    PipelineConfig, ProcessingResult, ProcessingStatus, VideoMetadata,
    TranslationJob, AudioGenerationJob, TTSEngine
)
from .exceptions import PipelineError, VideoScrapingError, TranslationError, TTSError
from ..config.validation import validate_video_url, validate_language_code


logger = structlog.get_logger(__name__)


class TranslationPipeline:
    """Main translation pipeline orchestrator."""
    
    def __init__(
        self,
        video_service: VideoScrapingService,
        translation_service: TranslationService,
        tts_service: TTSService,
        storage_service: StorageService,
        config: PipelineConfig
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            video_service: Video scraping service
            translation_service: Translation service
            tts_service: Text-to-speech service
            storage_service: Storage service
            config: Pipeline configuration
        """
        self.video_service = video_service
        self.translation_service = translation_service
        self.tts_service = tts_service
        self.storage_service = storage_service
        self.config = config
        
        logger.info(
            "Translation pipeline initialized",
            target_language=config.target_language,
            tts_engine=config.tts_engine.value,
            output_directory=str(config.output_directory)
        )
    
    async def process_video(self, url: str, video_id: Optional[str] = None) -> ProcessingResult:
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
            tts_engine=self.config.tts_engine
        )
        
        logger.info("Starting video processing", video_id=video_id, url=url)
        
        try:
            # Step 1: Scrape video and extract metadata/transcript
            metadata, transcript_segments = await self._scrape_video(url)
            result.metadata = metadata
            
            if not transcript_segments:
                result.mark_failed("No transcript available for translation")
                await self.storage_service.save_processing_result(result)
                return result
            
            # Step 2: Translate transcript segments
            translated_segments = await self._translate_segments(
                video_id, transcript_segments, self.config.target_language
            )
            
            # Step 3: Generate audio for translated segments
            audio_files = await self._generate_audio(
                video_id, translated_segments, self.config.target_language
            )
            
            # Step 4: Save all data to storage
            await self._save_pipeline_data(video_id, metadata, translated_segments, result)
            
            # Mark as completed
            result.mark_completed()
            
            logger.info(
                "Video processing completed",
                video_id=video_id,
                transcript_segments=len(transcript_segments),
                translated_segments=len(translated_segments),
                audio_files=len(audio_files)
            )
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error("Video processing failed", video_id=video_id, error=error_msg)
            result.mark_failed(error_msg)
        
        # Always save the result
        await self.storage_service.save_processing_result(result)
        return result
    
    async def process_video_batch(
        self, 
        urls: List[str], 
        max_concurrent: int = 3
    ) -> AsyncIterator[ProcessingResult]:
        """
        Process multiple videos concurrently.
        
        Args:
            urls: List of video URLs to process
            max_concurrent: Maximum concurrent processing tasks
            
        Yields:
            Processing results as videos are completed
        """
        logger.info("Starting batch processing", video_count=len(urls), max_concurrent=max_concurrent)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_video(url: str) -> ProcessingResult:
            async with semaphore:
                return await self.process_video(url)
        
        # Create tasks for all videos
        tasks = [process_single_video(url) for url in urls]
        
        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result
        
        logger.info("Batch processing completed", video_count=len(urls))
    
    async def process_existing_video(
        self, 
        video_id: str, 
        new_target_language: Optional[str] = None
    ) -> ProcessingResult:
        """
        Reprocess an existing video with new language or settings.
        
        Args:
            video_id: Existing video ID
            new_target_language: New target language (optional)
            
        Returns:
            Processing result
        """
        logger.info("Reprocessing existing video", video_id=video_id)
        
        # Load existing metadata
        metadata = await self.storage_service.load_metadata(video_id)
        if not metadata:
            raise PipelineError(f"Video metadata not found: {video_id}")
        
        # Use new language if provided
        target_language = new_target_language or self.config.target_language
        validate_language_code(target_language)
        
        # Initialize processing result
        result = ProcessingResult(
            video_id=video_id,
            status=ProcessingStatus.PROCESSING,
            metadata=metadata,
            target_language=target_language,
            tts_engine=self.config.tts_engine
        )
        
        try:
            # Get transcript from original video processing
            original_url = metadata.url
            _, transcript_segments = await self.video_service.get_transcript(original_url)
            
            if not transcript_segments:
                result.mark_failed("No transcript available for reprocessing")
                await self.storage_service.save_processing_result(result)
                return result
            
            # Translate and generate audio with new settings
            translated_segments = await self._translate_segments(
                video_id, transcript_segments, target_language
            )
            
            audio_files = await self._generate_audio(
                video_id, translated_segments, target_language
            )
            
            # Save updated data
            await self._save_pipeline_data(video_id, metadata, translated_segments, result)
            
            result.mark_completed()
            
            logger.info("Video reprocessing completed", video_id=video_id)
            
        except Exception as e:
            error_msg = f"Reprocessing failed: {str(e)}"
            logger.error("Video reprocessing failed", video_id=video_id, error=error_msg)
            result.mark_failed(error_msg)
        
        await self.storage_service.save_processing_result(result)
        return result
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get overall pipeline status and statistics.
        
        Returns:
            Dictionary with pipeline status information
        """
        try:
            # Get list of all videos
            video_ids = await self.storage_service.list_videos()
            
            status_counts = {
                ProcessingStatus.PENDING: 0,
                ProcessingStatus.PROCESSING: 0,
                ProcessingStatus.COMPLETED: 0,
                ProcessingStatus.FAILED: 0
            }
            
            processed_videos = []
            
            # Gather status for each video
            for video_id in video_ids:
                try:
                    # Try to load processing result to get status
                    video_dir = await self.storage_service.get_video_directory(video_id)
                    result_file = video_dir / "processing_result.json"
                    
                    if result_file.exists():
                        import json
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        status = ProcessingStatus(result_data['status'])
                        status_counts[status] += 1
                        
                        processed_videos.append({
                            "video_id": video_id,
                            "status": status.value,
                            "target_language": result_data.get('target_language'),
                            "completed_at": result_data.get('completed_at')
                        })
                    else:
                        # No processing result, assume pending
                        status_counts[ProcessingStatus.PENDING] += 1
                        processed_videos.append({
                            "video_id": video_id,
                            "status": ProcessingStatus.PENDING.value
                        })
                        
                except Exception as e:
                    logger.warning("Failed to get status for video", video_id=video_id, error=str(e))
                    continue
            
            # Get storage statistics
            storage_stats = await self.storage_service.get_storage_stats()
            
            return {
                "pipeline_config": {
                    "target_language": self.config.target_language,
                    "tts_engine": self.config.tts_engine.value,
                    "output_directory": str(self.config.output_directory)
                },
                "summary": {
                    "total_videos": len(video_ids),
                    "pending": status_counts[ProcessingStatus.PENDING],
                    "processing": status_counts[ProcessingStatus.PROCESSING],
                    "completed": status_counts[ProcessingStatus.COMPLETED],
                    "failed": status_counts[ProcessingStatus.FAILED]
                },
                "storage": storage_stats,
                "processed_videos": processed_videos
            }
            
        except Exception as e:
            logger.error("Failed to get pipeline status", error=str(e))
            raise PipelineError(f"Failed to get pipeline status: {str(e)}")
    
    async def _scrape_video(self, url: str) -> tuple[VideoMetadata, List]:
        """Scrape video and extract metadata and transcript."""
        try:
            logger.debug("Scraping video", url=url)
            metadata, transcript_segments = await self.video_service.scrape_video(url)
            
            logger.info(
                "Video scraped successfully",
                video_id=metadata.video_id,
                title=metadata.title,
                duration=metadata.duration,
                transcript_segments=len(transcript_segments)
            )
            
            return metadata, transcript_segments
            
        except Exception as e:
            logger.error("Video scraping failed", url=url, error=str(e))
            raise VideoScrapingError(f"Failed to scrape video: {str(e)}")
    
    async def _translate_segments(self, video_id: str, segments: List, target_language: str) -> List:
        """Translate transcript segments."""
        try:
            logger.debug(
                "Starting translation",
                video_id=video_id,
                segment_count=len(segments),
                target_language=target_language
            )
            
            # Create translation job
            job = TranslationJob(
                video_id=video_id,
                segments=segments,
                target_language=target_language,
                tts_engine=self.config.tts_engine,
                output_directory=await self.storage_service.get_video_directory(video_id)
            )
            
            # Translate segments
            translated_segments = []
            async for translated_segment in self.translation_service.translate_batch(job):
                translated_segments.append(translated_segment)
            
            logger.info(
                "Translation completed",
                video_id=video_id,
                translated_segments=len(translated_segments)
            )
            
            return translated_segments
            
        except Exception as e:
            logger.error("Translation failed", video_id=video_id, error=str(e))
            raise TranslationError(f"Failed to translate segments: {str(e)}")
    
    async def _generate_audio(self, video_id: str, segments: List, language: str) -> List[Path]:
        """Generate audio for translated segments."""
        try:
            logger.debug(
                "Starting audio generation",
                video_id=video_id,
                segment_count=len(segments),
                language=language
            )
            
            # Create audio generation job
            video_dir = await self.storage_service.get_video_directory(video_id)
            audio_dir = video_dir / "translated_audio"
            
            job = AudioGenerationJob(
                segments=segments,
                output_directory=audio_dir,
                language=language,
                tts_engine=self.config.tts_engine,
                audio_format=self.config.audio_format
            )
            
            # Generate audio files
            audio_files = []
            async for audio_path in self.tts_service.generate_batch_audio(job):
                audio_files.append(audio_path)
                
                # Update segment with audio path
                segment_index = len(audio_files) - 1
                if segment_index < len(segments):
                    segments[segment_index].audio_path = audio_path
            
            logger.info(
                "Audio generation completed",
                video_id=video_id,
                audio_files=len(audio_files)
            )
            
            return audio_files
            
        except Exception as e:
            logger.error("Audio generation failed", video_id=video_id, error=str(e))
            raise TTSError(f"Failed to generate audio: {str(e)}")
    
    async def _save_pipeline_data(
        self, 
        video_id: str, 
        metadata: VideoMetadata, 
        translated_segments: List, 
        result: ProcessingResult
    ) -> None:
        """Save all pipeline data to storage."""
        try:
            # Save metadata
            metadata_path = await self.storage_service.save_metadata(video_id, metadata)
            result.files["metadata"] = str(metadata_path)
            
            # Save translation data
            translation_path = await self.storage_service.save_translation_data(
                video_id, translated_segments
            )
            result.files["translation"] = str(translation_path)
            
            # Add audio directory to files
            video_dir = await self.storage_service.get_video_directory(video_id)
            audio_dir = video_dir / "translated_audio"
            if audio_dir.exists():
                result.files["audio_directory"] = str(audio_dir)
            
            logger.debug("Pipeline data saved", video_id=video_id)
            
        except Exception as e:
            logger.error("Failed to save pipeline data", video_id=video_id, error=str(e))
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