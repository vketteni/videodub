#!/usr/bin/env python3
"""
Test evaluation framework with real YouTube video data.

This script demonstrates end-to-end integration:
1. Use video-scraper to get real YouTube transcript
2. Convert to evaluation dataset format
3. Run transcript processing evaluation
4. Generate human-readable report

Usage:
    poetry run python3 evaluation/scripts/test_real_video.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import video-scraper
try:
    from video_scraper import VideoScraperAPI, scrape_metadata_only
    print("✅ video-scraper imported successfully")
except ImportError as e:
    print(f"❌ Failed to import video-scraper: {e}")
    sys.exit(1)

# Import evaluation framework
try:
    from evaluation.evaluators import (
        EvaluationConfig,
        EvaluationDataset,
        TranscriptProcessingEvaluator,
    )
    print("✅ Evaluation framework imported successfully")
except ImportError as e:
    print(f"❌ Failed to import evaluation framework: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_transcript_to_segments(transcript_entries: List) -> List[Dict]:
    """Convert video-scraper transcript entries to evaluation format."""
    segments = []
    
    for i, entry in enumerate(transcript_entries):
        segment = {
            "start": entry.start,
            "end": entry.start + entry.duration,
            "text": entry.text.strip(),
            "speaker": "Speaker"  # video-scraper doesn't provide speaker info
        }
        segments.append(segment)
    
    return segments


def create_evaluation_dataset(metadata, transcript_segments: List[Dict], max_segments: int = 10) -> EvaluationDataset:
    """Create an evaluation dataset from real video data."""
    
    # Limit segments for testing (full videos can be very long)
    limited_segments = transcript_segments[:max_segments]
    
    # Create sample description
    description = f"Real YouTube video: {metadata.title[:50]}..." if len(metadata.title) > 50 else f"Real YouTube video: {metadata.title}"
    
    # Create dataset
    dataset = EvaluationDataset(
        name="real_youtube_video",
        description="Evaluation using real YouTube video transcript data",
        input_samples=[
            {
                "segments": limited_segments,
                "description": description,
                "video_metadata": {
                    "title": metadata.title,
                    "duration": metadata.duration,
                    "channel": getattr(metadata, 'channel', 'Unknown'),
                    "url": getattr(metadata, 'url', 'Unknown')
                }
            }
        ],
        metadata={
            "source": "youtube",
            "video_title": metadata.title,
            "total_segments": len(transcript_segments),
            "evaluated_segments": len(limited_segments),
            "difficulty": "real_world",
            "created_at": datetime.now().isoformat()
        }
    )
    
    return dataset


async def main():
    """Main function to test evaluation with real video data."""
    parser = argparse.ArgumentParser(description="Test evaluation framework with real YouTube video")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--max-segments", type=int, default=10, help="Maximum number of segments to evaluate")
    parser.add_argument("--config", default="hybrid", choices=["rule_based", "ai_enhanced", "hybrid"], help="Processing configuration")
    parser.add_argument("--save-dataset", action="store_true", help="Save extracted dataset for future use")
    
    args = parser.parse_args()
    
    print(f"🎬 Testing evaluation framework with real YouTube video")
    print(f"📹 URL: {args.url}")
    print(f"⚙️  Config: {args.config}")
    print(f"📊 Max segments: {args.max_segments}")
    print("-" * 60)
    
    try:
        # Step 1: Scrape video metadata and transcript
        print("1️⃣ Scraping video metadata and transcript...")
        
        # Use video-scraper async API to avoid event loop conflicts
        scraper_api = VideoScraperAPI()
        scraping_result = await scraper_api.scraper.scrape(args.url)
        
        if not scraping_result.transcript:
            print("❌ No transcript found for this video")
            print("💡 Try a video with captions/subtitles enabled")
            return
        
        print(f"✅ Found {len(scraping_result.transcript)} transcript entries")
        print(f"📝 Video: {scraping_result.metadata.title}")
        print(f"⏱️  Duration: {scraping_result.metadata.duration}s")
        
        # Step 2: Convert to evaluation format
        print("\n2️⃣ Converting transcript to evaluation format...")
        
        transcript_segments = convert_transcript_to_segments(scraping_result.transcript)
        evaluation_dataset = create_evaluation_dataset(
            scraping_result.metadata, 
            transcript_segments, 
            args.max_segments
        )
        
        print(f"✅ Created evaluation dataset with {len(evaluation_dataset.input_samples[0]['segments'])} segments")
        
        # Display some sample segments
        print("\n📋 Sample transcript segments:")
        for i, seg in enumerate(evaluation_dataset.input_samples[0]['segments'][:3]):
            print(f"  {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text'][:60]}{'...' if len(seg['text']) > 60 else ''}\"")
        if len(evaluation_dataset.input_samples[0]['segments']) > 3:
            print(f"  ... and {len(evaluation_dataset.input_samples[0]['segments']) - 3} more segments")
        
        # Step 3: Save dataset if requested
        if args.save_dataset:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_path = Path(f"evaluation/datasets/transcript_processing/real_youtube_{timestamp}.json")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            evaluation_dataset.to_json(dataset_path)
            print(f"💾 Saved dataset to: {dataset_path}")
        
        # Step 4: Run evaluation
        print(f"\n3️⃣ Running transcript processing evaluation ({args.config} mode)...")
        
        evaluator = TranscriptProcessingEvaluator()
        config = EvaluationConfig(
            config_name=f"real_video_{args.config}",
            processing_mode=args.config,
            max_samples=1
        )
        
        results = await evaluator.evaluate_step(evaluation_dataset, config)
        
        # Step 5: Generate and display report
        print("\n4️⃣ Generating evaluation report...")
        
        report = evaluator.generate_report(results)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"evaluation/reports/generated/real_youtube_evaluation_{timestamp}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")
        
        # Display summary
        if results and results[0].error_info is None:
            result = results[0]
            print(f"\n📊 Evaluation Summary:")
            print(f"   Quality Score: {result.quality_score:.3f}")
            print(f"   Processing Time: {result.processing_time:.3f}s")
            print(f"   Input Segments: {result.metrics.get('input_segment_count', 'N/A')}")
            print(f"   Output Segments: {result.metrics.get('output_segment_count', 'N/A')}")
            print(f"   Segment Reduction: {result.metrics.get('segment_reduction_ratio', 0):.1%}")
            print(f"   Word Preservation: {result.metrics.get('word_preservation_ratio', 0):.1%}")
        else:
            print("❌ Evaluation failed")
            if results and results[0].error_info:
                print(f"   Error: {results[0].error_info}")
        
        print(f"\n🎉 End-to-end test completed!")
        print(f"📖 View full report: {report_path}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())