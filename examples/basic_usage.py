#!/usr/bin/env python3
"""Basic usage example for the YouTube Translation Pipeline."""

import asyncio
import os
from pathlib import Path

from pathlib import Path
from videodub import (
    TTSEngine,
    configure_logging,
    OpenAITranslationService,
    FallbackTranslationService,
    create_tts_service,
    create_audio_processing_service,
    FFmpegVideoProcessingService,
    FileStorageService,
)
from videodub.core.pipeline import TranslationPipeline
from videodub.services.data_extraction import YouTubeDataExtractionService
from videodub.services.alignment import TimingAlignmentService
from videodub.core.models import PipelineConfig


async def basic_translation_example():
    """Basic example: translate a single video to Spanish."""
    print("=== Basic Translation Example ===")

    # Set up logging
    configure_logging(level="INFO", enable_json=False)

    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create output directory
    output_path = Path("./example_output")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create services
    data_extraction = YouTubeDataExtractionService(output_dir=output_path)
    alignment = TimingAlignmentService()
    
    if openai_api_key:
        # Create pipeline with OpenAI services
        translation = OpenAITranslationService(api_key=openai_api_key)
        tts = create_tts_service(engine=TTSEngine.OPENAI, openai_api_key=openai_api_key)
        print("✅ Using OpenAI for translation and TTS")
    else:
        # Fallback to system TTS
        translation = FallbackTranslationService()
        tts = create_tts_service(engine=TTSEngine.SYSTEM)
        print("⚠️  No OpenAI API key found, using fallback translation and system TTS")
    
    # Create other services
    audio_processing = create_audio_processing_service()
    video_processing = FFmpegVideoProcessingService()
    storage = FileStorageService(base_path=output_path)
    
    # Create pipeline config
    config = PipelineConfig(
        target_language="es",  # Spanish
        tts_engine=TTSEngine.OPENAI if openai_api_key else TTSEngine.SYSTEM,
        output_directory=str(output_path),
    )
    
    # Create pipeline
    pipeline = TranslationPipeline(
        data_extraction_service=data_extraction,
        translation_service=translation,
        alignment_service=alignment,
        tts_service=tts,
        audio_processing_service=audio_processing,
        video_processing_service=video_processing,
        storage_service=storage,
        config=config
    )

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
    """Example using the new pipeline with simpler configuration."""
    print("\n=== Simple Pipeline Example ===")

    # Create simple pipeline (auto-detects API key)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create output directory
    output_path = Path("./simple_output")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create services with minimal configuration
    data_extraction = YouTubeDataExtractionService(output_dir=output_path)
    alignment = TimingAlignmentService()
    
    # Auto-detect best translation service
    translation = OpenAITranslationService(api_key=openai_api_key) if openai_api_key else FallbackTranslationService()
    tts = create_tts_service(engine=TTSEngine.OPENAI if openai_api_key else TTSEngine.SYSTEM, openai_api_key=openai_api_key)
    
    # Create pipeline
    pipeline = TranslationPipeline(
        data_extraction_service=data_extraction,
        translation_service=translation,
        alignment_service=alignment,
        tts_service=tts,
        audio_processing_service=create_audio_processing_service(),
        video_processing_service=FFmpegVideoProcessingService(),
        storage_service=FileStorageService(base_path=output_path),
        config=PipelineConfig(
            target_language="fr",  # French
            tts_engine=TTSEngine.OPENAI if openai_api_key else TTSEngine.SYSTEM,
            output_directory=str(output_path),
        )
    )

    # Test with a simple video
    try:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"🎬 Testing pipeline with: {video_url}")
        
        result = await pipeline.process_video(video_url)
        
        print(f"📈 Pipeline Result:")
        print(f"  Status: {result.status.value}")
        print(f"  Target Language: {result.target_language}")
        print(f"  TTS Engine: {result.tts_engine}")
        
        if result.files:
            print(f"  Generated Files: {len(result.files)}")

    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")


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

        # Create output directory for this language
        output_path = Path(f"./multi_lang_output/{lang}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline for this language
        pipeline = TranslationPipeline(
            data_extraction_service=YouTubeDataExtractionService(output_dir=output_path),
            translation_service=OpenAITranslationService(api_key=openai_api_key),
            alignment_service=TimingAlignmentService(),
            tts_service=create_tts_service(engine=TTSEngine.OPENAI, openai_api_key=openai_api_key),
            audio_processing_service=create_audio_processing_service(),
            video_processing_service=FFmpegVideoProcessingService(),
            storage_service=FileStorageService(base_path=output_path),
            config=PipelineConfig(
                target_language=lang,
                tts_engine=TTSEngine.OPENAI,
                output_directory=str(output_path),
            )
        )

        try:
            result = await pipeline.process_video(video_url)
            print(f"  ✅ {lang}: {result.status.value}")

        except Exception as e:
            print(f"  ❌ {lang}: {e}")


async def alignment_strategies_example():
    """Example showing different alignment strategies."""
    print("\n=== Alignment Strategies Example ===")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  Skipping alignment example (requires OpenAI API key)")
        return

    # Create output directory
    output_path = Path("./alignment_output")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline with alignment comparison support
    pipeline = TranslationPipeline(
        data_extraction_service=YouTubeDataExtractionService(output_dir=output_path),
        translation_service=OpenAITranslationService(api_key=openai_api_key),
        alignment_service=TimingAlignmentService(),
        tts_service=create_tts_service(engine=TTSEngine.OPENAI, openai_api_key=openai_api_key),
        audio_processing_service=create_audio_processing_service(),
        video_processing_service=FFmpegVideoProcessingService(),
        storage_service=FileStorageService(base_path=output_path),
        config=PipelineConfig(
            target_language="es",
            tts_engine=TTSEngine.OPENAI,
            output_directory=str(output_path),
        )
    )

    try:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"🎬 Testing alignment strategies with: {video_url}")
        
        # Process with alignment comparison to test multiple strategies
        result = await pipeline.process_video_with_alignment_comparison(video_url)
        
        print(f"📊 Alignment Comparison Results:")
        print(f"  Status: {result.status.value}")
        print(f"  Best Strategy: {result.alignment_evaluation.best_strategy if result.alignment_evaluation else 'N/A'}")
        
        if result.alignment_evaluation:
            print(f"  Strategies Tested: {len(result.alignment_evaluation.strategy_results)}")
            for strategy, eval_result in result.alignment_evaluation.strategy_results.items():
                print(f"    {strategy}: Score {eval_result.overall_score:.3f}")

    except Exception as e:
        print(f"❌ Alignment example error: {e}")


async def sequential_processing_example():
    """Example of processing multiple videos sequentially."""
    print("\n=== Sequential Processing Example ===")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  Skipping sequential example (requires OpenAI API key)")
        return

    # Multiple video URLs (using same URL for demo)
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
    ]

    print(f"🎬 Processing {len(video_urls)} videos sequentially...")

    results = []
    for i, url in enumerate(video_urls):
        print(f"\n📹 Processing video {i+1}/{len(video_urls)}: {url}")
        
        # Create output directory for each video
        output_path = Path(f"./sequential_output/video_{i+1}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline for this video
        pipeline = TranslationPipeline(
            data_extraction_service=YouTubeDataExtractionService(output_dir=output_path),
            translation_service=OpenAITranslationService(api_key=openai_api_key),
            alignment_service=TimingAlignmentService(),
            tts_service=create_tts_service(engine=TTSEngine.OPENAI, openai_api_key=openai_api_key),
            audio_processing_service=create_audio_processing_service(),
            video_processing_service=FFmpegVideoProcessingService(),
            storage_service=FileStorageService(base_path=output_path),
            config=PipelineConfig(
                target_language="es",
                tts_engine=TTSEngine.OPENAI,
                output_directory=str(output_path),
            )
        )
        
        try:
            result = await pipeline.process_video(url)
            results.append(result)
            print(f"  ✅ Video {i+1}: {result.status.value}")

        except Exception as e:
            print(f"  ❌ Video {i+1}: {e}")

    # Summary
    successful = len([r for r in results if r.status.value == "completed"])
    print(f"\n📊 Sequential Results: {successful}/{len(video_urls)} successful")


async def main():
    """Run all examples."""
    print("🚀 YouTube Translation Pipeline - Examples")
    print("=" * 50)

    try:
        await basic_translation_example()
        await simple_pipeline_example()
        await multi_language_example()
        await alignment_strategies_example()
        await sequential_processing_example()

        print("\n✅ All examples completed!")
        print("\n💡 Tips:")
        print("  - Set OPENAI_API_KEY environment variable for full functionality")
        print("  - Check the output directories for generated files")
        print("  - Use the Makefile for development tasks: make help")
        print("  - Examples now use the new TranslationPipeline with alignment services")

    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
