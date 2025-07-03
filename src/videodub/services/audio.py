"""Audio processing service for combining and manipulating audio files."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
import tempfile
import shutil

from ..core.interfaces import AudioProcessingService
from ..core.models import TranslationSegment
from ..core.exceptions import PipelineError

logger = logging.getLogger(__name__)


class FFmpegAudioProcessingService(AudioProcessingService):
    """Audio processing service using FFmpeg for audio manipulation."""
    
    def __init__(self):
        """Initialize the FFmpeg audio processor."""
        self._check_ffmpeg_availability()
    
    def _check_ffmpeg_availability(self) -> None:
        """Check if FFmpeg is available on the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                raise PipelineError("FFmpeg is not available or not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise PipelineError(f"FFmpeg is not installed or not accessible: {e}")
    
    async def combine_audio_segments(
        self, 
        audio_files: List[Path], 
        output_path: Path,
        segments: List[TranslationSegment]
    ) -> Path:
        """
        Combine multiple audio segments into a single file using FFmpeg.
        
        Args:
            audio_files: List of audio file paths in order
            output_path: Where to save combined audio
            segments: Translation segments with timing info
            
        Returns:
            Path to combined audio file
        """
        if not audio_files:
            raise PipelineError("No audio files provided for combination")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter out non-existent files
        existing_files = [f for f in audio_files if f.exists()]
        if not existing_files:
            raise PipelineError("No existing audio files found for combination")
        
        if len(existing_files) != len(audio_files):
            logger.warning(f"Some audio files missing: {len(audio_files) - len(existing_files)} files not found")
        
        logger.info(f"Combining {len(existing_files)} audio segments into {output_path}")
        
        try:
            # Create temporary file list for FFmpeg concat
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = Path(f.name)
                for audio_file in existing_files:
                    # Use absolute paths for FFmpeg
                    abs_path = audio_file.resolve()
                    # Escape paths for FFmpeg
                    escaped_path = str(abs_path).replace("'", "'\"'\"'")
                    f.write(f"file '{escaped_path}'\n")
            
            # Use FFmpeg concat demuxer for lossless concatenation
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",  # Copy without re-encoding for speed
                "-y",  # Overwrite output file
                str(output_path)
            ]
            
            # Run FFmpeg command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg failed: {error_msg}")
                raise PipelineError(f"Failed to combine audio files: {error_msg}")
            
            # Clean up temporary file
            concat_file.unlink(missing_ok=True)
            
            if not output_path.exists():
                raise PipelineError(f"Combined audio file was not created: {output_path}")
            
            logger.info(f"Successfully combined audio segments to {output_path}")
            return output_path
            
        except Exception as e:
            # Clean up on error
            if 'concat_file' in locals():
                concat_file.unlink(missing_ok=True)
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise PipelineError(f"Error combining audio segments: {e}")
    
    async def adjust_audio_timing(
        self, 
        audio_path: Path, 
        segment: TranslationSegment
    ) -> Path:
        """
        Adjust audio timing to match original segment duration.
        
        Args:
            audio_path: Path to audio file
            segment: Translation segment with timing info
            
        Returns:
            Path to adjusted audio file
        """
        if not audio_path.exists():
            raise PipelineError(f"Audio file not found: {audio_path}")
        
        # Calculate target duration
        target_duration = segment.end_time - segment.start_time
        if target_duration <= 0:
            logger.warning(f"Invalid segment duration: {target_duration}s")
            return audio_path
        
        # Create output path
        output_path = audio_path.parent / f"{audio_path.stem}_adjusted{audio_path.suffix}"
        
        logger.info(f"Adjusting audio timing for {audio_path} to {target_duration}s")
        
        try:
            # Use FFmpeg to adjust timing while preserving quality
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-af", f"atempo={target_duration}",  # Adjust tempo to match duration
                "-y",  # Overwrite output file
                str(output_path)
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                logger.warning(f"Audio timing adjustment failed: {error_msg}")
                # Return original file if adjustment fails
                return audio_path
            
            if not output_path.exists():
                logger.warning(f"Adjusted audio file was not created: {output_path}")
                return audio_path
            
            logger.info(f"Successfully adjusted audio timing: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"Error adjusting audio timing: {e}")
            # Return original file if adjustment fails
            return audio_path


class PydubAudioProcessingService(AudioProcessingService):
    """Fallback audio processing service using pydub when FFmpeg is not available."""
    
    def __init__(self):
        """Initialize the pydub audio processor."""
        try:
            import pydub
            self.pydub = pydub
        except ImportError:
            raise PipelineError("pydub is not installed. Please install it or use FFmpeg.")
    
    async def combine_audio_segments(
        self, 
        audio_files: List[Path], 
        output_path: Path,
        segments: List[TranslationSegment]
    ) -> Path:
        """
        Combine multiple audio segments using pydub.
        
        Args:
            audio_files: List of audio file paths in order
            output_path: Where to save combined audio
            segments: Translation segments with timing info
            
        Returns:
            Path to combined audio file
        """
        if not audio_files:
            raise PipelineError("No audio files provided for combination")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter out non-existent files
        existing_files = [f for f in audio_files if f.exists()]
        if not existing_files:
            raise PipelineError("No existing audio files found for combination")
        
        logger.info(f"Combining {len(existing_files)} audio segments using pydub")
        
        try:
            combined_audio = None
            
            for audio_file in existing_files:
                try:
                    audio_segment = self.pydub.AudioSegment.from_wav(str(audio_file))
                    if combined_audio is None:
                        combined_audio = audio_segment
                    else:
                        combined_audio += audio_segment
                except Exception as e:
                    logger.warning(f"Failed to load audio file {audio_file}: {e}")
                    continue
            
            if combined_audio is None:
                raise PipelineError("No audio segments could be loaded")
            
            # Export combined audio
            combined_audio.export(str(output_path), format="wav")
            
            if not output_path.exists():
                raise PipelineError(f"Combined audio file was not created: {output_path}")
            
            logger.info(f"Successfully combined audio segments to {output_path}")
            return output_path
            
        except Exception as e:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise PipelineError(f"Error combining audio segments with pydub: {e}")
    
    async def adjust_audio_timing(
        self, 
        audio_path: Path, 
        segment: TranslationSegment
    ) -> Path:
        """
        Adjust audio timing using pydub.
        
        Args:
            audio_path: Path to audio file
            segment: Translation segment with timing info
            
        Returns:
            Path to adjusted audio file (same as input for pydub implementation)
        """
        # Pydub timing adjustment is more complex, so we skip it for now
        logger.info("Audio timing adjustment not implemented for pydub processor")
        return audio_path


def create_audio_processing_service() -> AudioProcessingService:
    """
    Factory function to create the best available audio processing service.
    
    Returns:
        AudioProcessingService instance (FFmpeg preferred, pydub fallback)
    """
    try:
        return FFmpegAudioProcessingService()
    except PipelineError:
        logger.warning("FFmpeg not available, falling back to pydub")
        try:
            return PydubAudioProcessingService()
        except PipelineError:
            raise PipelineError(
                "No audio processing service available. Please install FFmpeg or pydub."
            )