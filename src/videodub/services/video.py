"""Video processing service for creating dubbed videos."""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from ..core.interfaces import VideoProcessingService
from ..core.models import TranslationSegment
from ..core.exceptions import PipelineError

logger = logging.getLogger(__name__)


class FFmpegVideoProcessingService(VideoProcessingService):
    """Video processing service using FFmpeg for video-audio merging."""
    
    def __init__(self):
        """Initialize the FFmpeg video processor."""
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
    
    async def create_dubbed_video(
        self, 
        original_video_path: Path, 
        translated_audio_path: Path, 
        output_path: Path,
        segments: List[TranslationSegment]
    ) -> Path:
        """
        Create dubbed video by combining original video with translated audio.
        
        Args:
            original_video_path: Path to original video file
            translated_audio_path: Path to translated audio file
            output_path: Where to save dubbed video
            segments: Translation segments with timing info
            
        Returns:
            Path to dubbed video file
        """
        # Validate input files
        if not original_video_path.exists():
            raise PipelineError(f"Original video file not found: {original_video_path}")
        
        if not translated_audio_path.exists():
            raise PipelineError(f"Translated audio file not found: {translated_audio_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating dubbed video: {original_video_path} + {translated_audio_path} -> {output_path}")
        
        try:
            # Use FFmpeg to replace audio track with translated audio
            cmd = [
                "ffmpeg",
                "-i", str(original_video_path),     # Input video
                "-i", str(translated_audio_path),   # Input audio
                "-c:v", "copy",                     # Copy video stream without re-encoding
                "-c:a", "aac",                      # Encode audio as AAC
                "-map", "0:v:0",                    # Map video from first input
                "-map", "1:a:0",                    # Map audio from second input
                "-shortest",                        # End output when shortest stream ends
                "-y",                               # Overwrite output file
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
                logger.error(f"FFmpeg dubbing failed: {error_msg}")
                raise PipelineError(f"Failed to create dubbed video: {error_msg}")
            
            if not output_path.exists():
                raise PipelineError(f"Dubbed video file was not created: {output_path}")
            
            # Verify the output file is valid
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise PipelineError(f"Dubbed video file is empty: {output_path}")
            
            logger.info(f"Successfully created dubbed video: {output_path} ({file_size:,} bytes)")
            return output_path
            
        except Exception as e:
            # Clean up on error
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise PipelineError(f"Error creating dubbed video: {e}")
    
    async def extract_video_metadata(self, video_path: Path) -> Dict:
        """
        Extract metadata from video file using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        if not video_path.exists():
            raise PipelineError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting video metadata from: {video_path}")
        
        try:
            # Use FFprobe to get video information
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFprobe error"
                logger.error(f"FFprobe failed: {error_msg}")
                raise PipelineError(f"Failed to extract video metadata: {error_msg}")
            
            # Parse JSON output
            metadata = json.loads(stdout.decode())
            
            # Extract key information
            format_info = metadata.get("format", {})
            video_streams = [s for s in metadata.get("streams", []) if s.get("codec_type") == "video"]
            audio_streams = [s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"]
            
            # Build simplified metadata
            video_metadata = {
                "file_path": str(video_path),
                "file_size": int(format_info.get("size", 0)),
                "duration": float(format_info.get("duration", 0)),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "format_name": format_info.get("format_name", "unknown"),
                "video_streams": len(video_streams),
                "audio_streams": len(audio_streams),
            }
            
            # Add video stream info if available
            if video_streams:
                video_stream = video_streams[0]  # Take first video stream
                video_metadata.update({
                    "video_codec": video_stream.get("codec_name", "unknown"),
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "fps": self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
                    "video_bitrate": int(video_stream.get("bit_rate", 0)),
                })
            
            # Add audio stream info if available
            if audio_streams:
                audio_stream = audio_streams[0]  # Take first audio stream
                video_metadata.update({
                    "audio_codec": audio_stream.get("codec_name", "unknown"),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                    "audio_bitrate": int(audio_stream.get("bit_rate", 0)),
                })
            
            logger.info(f"Extracted metadata: {video_metadata['duration']:.2f}s, {video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}")
            return video_metadata
            
        except json.JSONDecodeError as e:
            raise PipelineError(f"Failed to parse video metadata: {e}")
        except Exception as e:
            raise PipelineError(f"Error extracting video metadata: {e}")
    
    def _parse_fps(self, fps_string: str) -> float:
        """Parse FPS from FFprobe format (e.g., '25/1' -> 25.0)."""
        try:
            if "/" in fps_string:
                numerator, denominator = fps_string.split("/")
                return float(numerator) / float(denominator)
            return float(fps_string)
        except (ValueError, ZeroDivisionError):
            return 0.0


class BasicVideoProcessor(VideoProcessingService):
    """Basic video processor with minimal functionality."""
    
    async def create_dubbed_video(
        self, 
        original_video_path: Path, 
        translated_audio_path: Path, 
        output_path: Path,
        segments: List[TranslationSegment]
    ) -> Path:
        """Basic implementation - just copy the original video."""
        logger.warning("BasicVideoProcessor: No actual dubbing performed, copying original video")
        
        if not original_video_path.exists():
            raise PipelineError(f"Original video file not found: {original_video_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Just copy the original video (no dubbing)
        import shutil
        shutil.copy2(original_video_path, output_path)
        
        logger.info(f"Copied original video to: {output_path}")
        return output_path
    
    async def extract_video_metadata(self, video_path: Path) -> Dict:
        """Basic metadata extraction."""
        if not video_path.exists():
            raise PipelineError(f"Video file not found: {video_path}")
        
        # Return minimal metadata
        file_size = video_path.stat().st_size
        return {
            "file_path": str(video_path),
            "file_size": file_size,
            "duration": 0.0,  # Unknown
            "format_name": "unknown",
            "video_streams": 1,
            "audio_streams": 1,
        }


def create_video_processor() -> VideoProcessingService:
    """
    Factory function to create the best available video processor.
    
    Returns:
        VideoProcessingService instance (FFmpeg preferred, Basic fallback)
    """
    try:
        return FFmpegVideoProcessingService()
    except PipelineError:
        logger.warning("FFmpeg not available for video processing, using basic processor")
        return BasicVideoProcessor()