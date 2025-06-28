#!/usr/bin/env python3
"""Basic usage example for the YouTube Translation Pipeline."""

import asyncio
import os
from pathlib import Path

from youtube_translator import (
    create_pipeline,
    create_simple_pipeline,
    TTSEngine,
    configure_logging,
)


async def basic_translation_example():
    """Basic example: translate a single video to Spanish."""
    print("=== Basic Translation Example ===")
    
    # Set up logging
    configure_logging(level="INFO", enable_json=False)
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        # Create pipeline with OpenAI services
        pipeline = create_pipeline(
            output_directory="./example_output",
            target_language="es",  # Spanish
            tts_engine=TTSEngine.OPENAI,
            openai_api_key=openai_api_key
        )
        print("✅ Using OpenAI for translation and TTS")
    else:
        # Fallback to system TTS
        pipeline = create_pipeline(
            output_directory="./example_output",
            target_language="es",
            tts_engine=TTSEngine.SYSTEM
        )
        print("⚠️  No OpenAI API key found, using fallback translation and system TTS")
    
    # Process a video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        print(f"🎬 Processing video: {video_url}")
        result = await pipeline.process_video(video_url)
        
        print(f"\n📊 Results:")
        print(f"  Video ID: {result.video_id}")
        print(f"  Status: {result.status.value}")
        print(f"  Target Language: {result.target_language}")
        
        if result.metadata:
            print(f"  Title: {result.metadata.title}")
            print(f"  Duration: {result.metadata.duration} seconds")
        
        if result.files:
            print(f"  Generated Files:")
            for file_type, file_path in result.files.items():
                print(f"    {file_type}: {file_path}")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")


async def simple_pipeline_example():
    """Example using the simple pipeline factory."""
    print("\n=== Simple Pipeline Example ===")
    
    # Create simple pipeline (auto-detects API key)
    pipeline = create_simple_pipeline(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Get pipeline status
    try:
        status = await pipeline.get_pipeline_status()
        
        print(f"📈 Pipeline Status:")
        print(f"  Target Language: {status['pipeline_config']['target_language']}")
        print(f"  TTS Engine: {status['pipeline_config']['tts_engine']}")
        print(f"  Total Videos: {status['summary']['total_videos']}")
        print(f"  Completed: {status['summary']['completed']}")
        print(f"  Failed: {status['summary']['failed']}")
        
        if status['storage']:
            print(f"  Storage: {status['storage']['total_size_mb']} MB")
        
    except Exception as e:
        print(f"❌ Error getting status: {e}")


async def multi_language_example():
    """Example translating to multiple languages."""
    print("\n=== Multi-Language Example ===")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  Skipping multi-language example (requires OpenAI API key)")
        return
    
    languages = ["es", "fr", "de"]  # Spanish, French, German
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    for lang in languages:
        print(f"\n🌍 Translating to {lang}...")
        
        pipeline = create_pipeline(
            output_directory=f"./multi_lang_output/{lang}",
            target_language=lang,
            tts_engine=TTSEngine.OPENAI,
            openai_api_key=openai_api_key
        )
        
        try:
            result = await pipeline.process_video(video_url)
            print(f"  ✅ {lang}: {result.status.value}")
            
        except Exception as e:
            print(f"  ❌ {lang}: {e}")


async def reprocess_existing_video_example():
    """Example of reprocessing an existing video with new language."""
    print("\n=== Reprocess Existing Video Example ===")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  Skipping reprocess example (requires OpenAI API key)")
        return
    
    pipeline = create_pipeline(
        output_directory="./example_output",
        target_language="es",
        tts_engine=TTSEngine.OPENAI,
        openai_api_key=openai_api_key
    )
    
    # Assume we have a previously processed video
    try:
        # Get list of existing videos
        status = await pipeline.get_pipeline_status()
        processed_videos = status.get('processed_videos', [])
        
        if processed_videos:
            video_id = processed_videos[0]['video_id']
            print(f"🔄 Reprocessing video {video_id} to French...")
            
            result = await pipeline.process_existing_video(
                video_id=video_id,
                new_target_language="fr"
            )
            
            print(f"  ✅ Reprocessing: {result.status.value}")
        else:
            print("  ℹ️  No existing videos found to reprocess")
            
    except Exception as e:
        print(f"  ❌ Error reprocessing: {e}")


async def batch_processing_example():
    """Example of processing multiple videos in batch."""
    print("\n=== Batch Processing Example ===")
    
    pipeline = create_simple_pipeline(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Multiple video URLs (using same URL for demo)
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
    ]
    
    print(f"🎬 Processing {len(video_urls)} videos...")
    
    results = []
    try:
        async for result in pipeline.process_video_batch(video_urls, max_concurrent=2):
            results.append(result)
            print(f"  📹 {result.video_id}: {result.status.value}")
    
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    # Summary
    successful = len([r for r in results if r.status.value == "completed"])
    print(f"\n📊 Batch Results: {successful}/{len(video_urls)} successful")


async def main():
    """Run all examples."""
    print("🚀 YouTube Translation Pipeline - Examples")
    print("=" * 50)
    
    try:
        await basic_translation_example()
        await simple_pipeline_example()
        await multi_language_example()
        await reprocess_existing_video_example()
        await batch_processing_example()
        
        print("\n✅ All examples completed!")
        print("\n💡 Tips:")
        print("  - Set OPENAI_API_KEY environment variable for full functionality")
        print("  - Check the output directories for generated files")
        print("  - Use the Makefile for development tasks: make help")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())