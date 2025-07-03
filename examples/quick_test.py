#!/usr/bin/env python3
"""Quick test script for single video translation with model comparison."""

import asyncio
import os
import time
from pathlib import Path

from videodub import create_pipeline, TTSEngine, configure_logging
from videodub.utils.cost_tracking import get_session_cost_summary, reset_global_cost_tracker


async def quick_translation_test(
    video_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    target_language: str = "de",
    openai_model: str = "gpt-4.1-nano",
    tts_engine: TTSEngine = TTSEngine.OPENAI
):
    """
    Quick test for single video translation with timing and cost tracking.
    
    Args:
        video_url: YouTube video URL to translate
        target_language: Target language code (e.g., 'es', 'fr', 'de')
        openai_model: OpenAI model to use for translation
        tts_engine: TTS engine to use
    """
    print(f"🚀 Quick Translation Test")
    print(f"   Video: {video_url}")
    print(f"   Language: {target_language}")
    print(f"   Model: {openai_model}")
    print(f"   TTS: {tts_engine.value}")
    print("-" * 50)
    
    # Configure minimal logging
    configure_logging(level="INFO", enable_json=False)
    
    # Reset cost tracking for this test
    reset_global_cost_tracker()
    
    # Get API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return None
    
    # Create pipeline with test-specific settings
    pipeline = create_pipeline(
        output_directory=f"./quick_test_output/{openai_model.replace('.', '_')}",
        target_language=target_language,
        tts_engine=tts_engine,
        openai_api_key=openai_api_key,
        translation_model=openai_model
    )
    
    start_time = time.time()
    
    try:
        # Process video
        result = await pipeline.process_video(video_url)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Status: {result.status.value}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Video ID: {result.video_id}")
        
        if result.metadata:
            print(f"   Title: {result.metadata.title}")
            print(f"   Video Duration: {result.metadata.duration}s")
        
        # Get cost data from result or session tracking
        cost_summary = result.cost_summary or get_session_cost_summary()
        print(f"\n💰 Cost Analysis:")
        print(f"   Total Cost: ${cost_summary['totals']['total_cost']:.6f}")
        print(f"   Total Tokens: {cost_summary['totals']['total_tokens']:,}")
        print(f"   Total Requests: {cost_summary['totals']['total_requests']}")
        
        # Detailed breakdown by model
        for model_name, model_data in cost_summary['models'].items():
            print(f"\n   📱 {model_name}:")
            usage = model_data['usage']
            costs = model_data['costs']
            if usage['input_tokens'] > 0:
                print(f"      Input tokens: {usage['input_tokens']:,} (${costs['input_cost']:.6f})")
                print(f"      Output tokens: {usage['output_tokens']:,} (${costs['output_cost']:.6f})")
            if usage['characters'] > 0:
                print(f"      Characters: {usage['characters']:,} (${costs['audio_cost']:.6f})")
            print(f"      Requests: {usage['requests']}")
            print(f"      Model total: ${costs['total_cost']:.6f}")
        
        if result.files:
            print(f"\n📁 Generated Files:")
            for file_type, file_path in result.files.items():
                size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                print(f"   {file_type}: {file_path} ({size/1024:.1f}KB)")
        
        if result.errors:
            print(f"\n⚠️  Errors: {result.errors}")
        
        return {
            "model": openai_model,
            "language": target_language,
            "duration": duration,
            "status": result.status.value,
            "cost_summary": cost_summary,
            "video_id": result.video_id
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Error after {duration:.1f}s: {e}")
        return {
            "model": openai_model,
            "language": target_language,
            "duration": duration,
            "status": "failed",
            "error": str(e)
        }


async def model_comparison_test(
    video_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    target_language: str = "es",
    models: list = None
):
    """
    Compare different OpenAI models for translation economics.
    
    Args:
        video_url: YouTube video URL to translate
        target_language: Target language code
        models: List of OpenAI models to compare
    """
    if models is None:
        models = ["gpt-4.1-nano"]
    
    print(f"🔬 Model Comparison Test")
    print(f"   Models: {', '.join(models)}")
    print(f"   Video: {video_url}")
    print(f"   Language: {target_language}")
    print("=" * 50)
    
    results = []
    
    for model in models:
        print(f"\n🧪 Testing {model}...")
        result = await quick_translation_test(
            video_url=video_url,
            target_language=target_language,
            openai_model=model
        )
        if result:
            results.append(result)
    
    # Summary comparison
    print(f"\n📈 Comparison Summary:")
    print(f"{'Model':<15} {'Duration':<10} {'Status':<10} {'Total Cost':<12} {'Tokens':<10}")
    print("-" * 70)
    
    for result in results:
        duration = f"{result['duration']:.1f}s"
        status = result['status']
        
        if result.get('cost_summary'):
            cost = result['cost_summary']['totals']['total_cost']
            tokens = result['cost_summary']['totals']['total_tokens']
            cost_str = f"${cost:.6f}"
            tokens_str = f"{tokens:,}"
        else:
            cost_str = "N/A"
            tokens_str = "N/A"
        
        print(f"{result['model']:<15} {duration:<10} {status:<10} {cost_str:<12} {tokens_str:<10}")
    
    return results


async def main():
    """Run quick tests based on command line args or defaults."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "compare":
            # Model comparison mode
            await model_comparison_test()
        elif command == "quick":
            # Single quick test
            video_url = sys.argv[2] if len(sys.argv) > 2 else "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            await quick_translation_test(video_url=video_url)
        else:
            print("Usage:")
            print("  python quick_test.py quick [video_url]    # Quick single test")
            print("  python quick_test.py compare              # Compare models")
    else:
        # Default: quick test
        await quick_translation_test()


if __name__ == "__main__":
    asyncio.run(main())