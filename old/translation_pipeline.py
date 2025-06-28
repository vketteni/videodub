#!/usr/bin/env python3
"""
YouTube Video Translation Pipeline

This script creates a data pipeline that:
1. Scrapes YouTube videos and extracts audio
2. Translates the transcript to a target language
3. Generates new audio in the translated language
4. Outputs the translated media and metadata
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from video_scraper import VideoProcessor
from audio_translator import AudioTranslator

class TranslationPipeline:
    def __init__(self, 
                 output_dir: str = "./pipeline_output",
                 target_language: str = "es",
                 tts_engine: str = "openai",
                 audio_only: bool = False):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.target_language = target_language
        self.audio_only = audio_only
        
        # Initialize components
        self.video_processor = VideoProcessor(
            output_dir=str(self.output_dir / "scraped"),
            audio_format="mp3"
        )
        self.audio_translator = AudioTranslator(
            target_language=target_language,
            tts_engine=tts_engine
        )
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def process_video(self, url: str, video_id: str = None) -> Optional[Dict[str, Any]]:
        """Process a single video through the complete pipeline"""
        
        if not video_id:
            video_id = self._extract_video_id(url)
        
        self.logger.info(f"Starting pipeline for video: {video_id}")
        
        # Create output directory for this video
        video_output_dir = self.output_dir / f"video_{video_id}"
        video_output_dir.mkdir(exist_ok=True)
        
        pipeline_result = {
            "video_id": video_id,
            "url": url,
            "target_language": self.target_language,
            "processing_started": datetime.now().isoformat(),
            "status": "processing",
            "files": {},
            "errors": []
        }
        
        try:
            # Step 1: Scrape video/audio
            self.logger.info("Step 1: Scraping video content...")
            if self.audio_only:
                scraped_data = self.video_processor.scrape_audio_only(url)
            else:
                scraped_data = self.video_processor.scrape_video(url)
            
            if not scraped_data:
                raise Exception("Failed to scrape video content")
            
            pipeline_result["title"] = scraped_data["title"]
            pipeline_result["duration"] = scraped_data.get("duration", 0)
            
            # Find the actual audio file from the scraper output directory
            audio_file = self._find_audio_file(video_id)
            pipeline_result["files"]["original_audio"] = audio_file or scraped_data.get("audio_path") or scraped_data.get("file_path")
            
            # Step 2: Process transcript if available
            if "transcript" in scraped_data and scraped_data["transcript"]:
                self.logger.info("Step 2: Translating transcript...")
                
                translated_segments = self.audio_translator.translate_transcript(
                    scraped_data["transcript"],
                    self.target_language
                )
                
                # Save translation data
                translation_file = video_output_dir / "translation.json"
                self.audio_translator.save_translation_data(translated_segments, str(translation_file))
                pipeline_result["files"]["translation"] = str(translation_file)
                
                # Step 3: Generate translated audio
                self.logger.info("Step 3: Generating translated audio...")
                audio_output_dir = video_output_dir / "translated_audio"
                translated_audio_path = self.audio_translator.generate_translated_audio(
                    translated_segments,
                    str(audio_output_dir)
                )
                pipeline_result["files"]["translated_audio"] = translated_audio_path
                
            else:
                self.logger.warning("No transcript available for translation")
                pipeline_result["errors"].append("No transcript available")
            
            # Step 4: Save metadata
            metadata_file = video_output_dir / "metadata.json"
            scraped_data["pipeline_info"] = pipeline_result
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=2, ensure_ascii=False)
            
            pipeline_result["files"]["metadata"] = str(metadata_file)
            pipeline_result["status"] = "completed"
            pipeline_result["processing_completed"] = datetime.now().isoformat()
            
            self.logger.info(f"Pipeline completed successfully for video: {video_id}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for video {video_id}: {str(e)}")
            pipeline_result["status"] = "failed"
            pipeline_result["errors"].append(str(e))
            pipeline_result["processing_completed"] = datetime.now().isoformat()
        
        # Save pipeline result (convert Path objects to strings for JSON serialization)
        result_file = video_output_dir / "pipeline_result.json"
        
        # Convert any Path objects to strings for JSON serialization
        def convert_paths_to_strings(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths_to_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            else:
                return obj
        
        serializable_result = convert_paths_to_strings(pipeline_result)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        return pipeline_result
    
    def process_video_list(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple videos through the pipeline"""
        results = []
        
        for i, url in enumerate(urls, 1):
            self.logger.info(f"Processing video {i}/{len(urls)}: {url}")
            result = self.process_video(url)
            results.append(result)
        
        # Save batch results
        batch_file = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from URL for naming purposes"""
        try:
            if "youtube.com" in url or "youtu.be" in url:
                if "v=" in url:
                    return url.split("v=")[1].split("&")[0]
                elif "youtu.be/" in url:
                    return url.split("youtu.be/")[1].split("?")[0]
            return f"video_{hash(url) % 10000:04d}"
        except:
            return f"video_{hash(url) % 10000:04d}"
    
    def _find_audio_file(self, video_id: str) -> Optional[str]:
        """Find the actual audio file that was downloaded"""
        try:
            scraped_dir = self.output_dir / "scraped" / video_id
            if scraped_dir.exists():
                # Look for audio files (mp3, m4a, wav, etc.)
                for audio_ext in ['.mp3', '.m4a', '.wav', '.aac', '.ogg']:
                    audio_files = list(scraped_dir.glob(f"*{audio_ext}"))
                    if audio_files:
                        return str(audio_files[0])  # Return first match
            return None
        except Exception as e:
            self.logger.error(f"Error finding audio file for {video_id}: {str(e)}")
            return None
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status and statistics"""
        status = {
            "output_directory": str(self.output_dir),
            "target_language": self.target_language,
            "processed_videos": [],
            "summary": {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "processing": 0
            }
        }
        
        # Scan output directory for processed videos
        for video_dir in self.output_dir.glob("video_*"):
            if video_dir.is_dir():
                result_file = video_dir / "pipeline_result.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result = json.load(f)
                        status["processed_videos"].append(result)
                        status["summary"]["total"] += 1
                        status["summary"][result["status"]] += 1
                    except:
                        pass
        
        return status

def main():
    parser = argparse.ArgumentParser(description="YouTube Video Translation Pipeline")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs to process")
    parser.add_argument("--output-dir", "-o", default="./pipeline_output", 
                       help="Output directory for processed videos")
    parser.add_argument("--target-language", "-l", default="es", 
                       help="Target language code (e.g., es, fr, de, ja)")
    parser.add_argument("--tts-engine", choices=["openai", "google", "azure", "system"], 
                       default="openai", help="Text-to-speech engine to use")
    parser.add_argument("--audio-only", action="store_true", 
                       help="Extract audio only (faster processing)")
    parser.add_argument("--status", action="store_true", 
                       help="Show pipeline status and exit")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TranslationPipeline(
        output_dir=args.output_dir,
        target_language=args.target_language,
        tts_engine=args.tts_engine,
        audio_only=args.audio_only
    )
    
    if args.status:
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2))
        return
    
    # Process videos
    if len(args.urls) == 1:
        result = pipeline.process_video(args.urls[0])
        print(f"\nPipeline completed for video: {result['video_id']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"Output directory: {args.output_dir}/video_{result['video_id']}")
    else:
        results = pipeline.process_video_list(args.urls)
        completed = sum(1 for r in results if r['status'] == 'completed')
        print(f"\nBatch processing completed: {completed}/{len(results)} videos successful")

if __name__ == "__main__":
    main()