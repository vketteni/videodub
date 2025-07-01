import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from video_scraper import VideoScraperAPI, scrape_audio_only

class VideoProcessor:
    def __init__(self, output_dir: str = "./downloads", audio_format: str = "mp3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audio_format = audio_format
        self.scraper = VideoScraperAPI(
            output_dir=str(self.output_dir),
            extract_audio=True
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def scrape_video(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape video and extract audio with metadata"""
        try:
            self.logger.info(f"Scraping video: {url}")
            result = self.scraper.scrape(url)
            
            if not result.success:
                self.logger.error(f"Failed to scrape video: {result.error}")
                return None
            
            video_data = {
                "video_id": getattr(result, 'video_id', None) or getattr(result.metadata, 'video_id', None),
                "url": url,
                "title": result.metadata.title,
                "duration": result.metadata.duration,
                "file_path": getattr(result, 'file_path', None),
                "audio_path": getattr(result, 'audio_path', None),
                "video_path": getattr(result, 'video_path', None),
                "metadata": {
                    "channel": getattr(result.metadata, 'channel', None),
                    "upload_date": getattr(result.metadata, 'upload_date', None),
                    "view_count": getattr(result.metadata, 'view_count', None),
                    "description": getattr(result.metadata, 'description', None)
                }
            }
            
            # Check if transcript exists in metadata
            if hasattr(result.metadata, 'transcript') and result.metadata.transcript:
                video_data["transcript"] = [
                    {"start": entry.start, "text": entry.text} 
                    for entry in result.metadata.transcript
                ]
            else:
                # Try to load transcript from the metadata.json file if it exists
                try:
                    import json
                    scraped_dir = self.output_dir / result.metadata.get('video_id', 'unknown')
                    metadata_file = scraped_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        if 'transcript' in metadata:
                            video_data["transcript"] = metadata['transcript']
                except Exception as e:
                    self.logger.warning(f"Could not load transcript from metadata: {str(e)}")
            
            self.logger.info(f"Successfully scraped: {result.metadata.title}")
            return video_data
            
        except Exception as e:
            self.logger.error(f"Error scraping video {url}: {str(e)}")
            return None
    
    def scrape_audio_only(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape audio only for faster processing"""
        try:
            self.logger.info(f"Scraping audio from: {url}")
            result = scrape_audio_only(url)
            
            if not result.success:
                self.logger.error(f"Failed to scrape audio: {result.error}")
                return None
            
            audio_data = {
                "url": url,
                "title": result.metadata.title,
                "duration": result.metadata.duration,
                "audio_path": getattr(result, 'file_path', None) or getattr(result, 'audio_path', None),
                "format": self.audio_format
            }
            
            self.logger.info(f"Successfully extracted audio: {result.metadata.title}")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Error extracting audio from {url}: {str(e)}")
            return None
    
    def save_metadata(self, video_data: Dict[str, Any], filename: str = None) -> str:
        """Save video metadata to JSON file"""
        if not filename:
            safe_title = "".join(c for c in video_data["title"][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_title}_metadata.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Metadata saved to: {filepath}")
        return str(filepath)