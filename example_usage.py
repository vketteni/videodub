#!/usr/bin/env python3
"""
Example usage of the YouTube Translation Pipeline
"""

from translation_pipeline import TranslationPipeline
from config import Config

def example_single_video():
    """Example: Process a single YouTube video"""
    print("=== Single Video Processing Example ===")
    
    # Initialize pipeline
    pipeline = TranslationPipeline(
        output_dir="./example_output",
        target_language="es",  # Spanish
        tts_engine="system",   # Use system TTS (no API key required)
        audio_only=True        # Faster processing
    )
    
    # Process a video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = pipeline.process_video(video_url)
    
    print(f"Video ID: {result['video_id']}")
    print(f"Title: {result.get('title', 'Unknown')}")
    print(f"Status: {result['status']}")
    print(f"Files generated: {list(result.get('files', {}).keys())}")
    
    if result['errors']:
        print(f"Errors: {result['errors']}")

def example_batch_processing():
    """Example: Process multiple videos"""
    print("\n=== Batch Processing Example ===")
    
    pipeline = TranslationPipeline(
        output_dir="./batch_output",
        target_language="fr",  # French
        tts_engine="system"
    )
    
    # Multiple video URLs
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        # Add more URLs as needed
    ]
    
    results = pipeline.process_video_list(video_urls)
    
    successful = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Processed {len(results)} videos")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

def example_with_openai():
    """Example: Using OpenAI for translation and TTS (requires API key)"""
    print("\n=== OpenAI Integration Example ===")
    
    # Check if OpenAI API key is available
    if not Config.OPENAI_API_KEY:
        print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return
    
    pipeline = TranslationPipeline(
        output_dir="./openai_output",
        target_language="ja",  # Japanese
        tts_engine="openai"
    )
    
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = pipeline.process_video(video_url)
    
    print(f"Processing with OpenAI TTS: {result['status']}")

def example_pipeline_status():
    """Example: Check pipeline status"""
    print("\n=== Pipeline Status Example ===")
    
    pipeline = TranslationPipeline(output_dir="./example_output")
    status = pipeline.get_pipeline_status()
    
    print(f"Output directory: {status['output_directory']}")
    print(f"Target language: {status['target_language']}")
    print(f"Total processed: {status['summary']['total']}")
    print(f"Completed: {status['summary']['completed']}")
    print(f"Failed: {status['summary']['failed']}")
    
    if status['processed_videos']:
        print("\nProcessed videos:")
        for video in status['processed_videos']:
            print(f"  - {video['video_id']}: {video['status']}")

def main():
    """Run all examples"""
    print("YouTube Translation Pipeline - Examples")
    print("=====================================")
    
    try:
        # Run examples
        example_single_video()
        example_pipeline_status()
        
        # Uncomment to run additional examples
        # example_batch_processing()
        # example_with_openai()
        
        print("\n=== Examples completed ===")
        print("Check the output directories for generated files.")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have installed the requirements:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()