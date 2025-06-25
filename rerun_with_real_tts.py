#!/usr/bin/env python3
"""
Clean up empty audio files and re-run with proper OpenAI TTS
"""

import json
import os
import shutil
from pathlib import Path
from audio_translator import AudioTranslator
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def cleanup_and_rerun(video_id="8XdL-7tAqnU", target_language="de"):
    """Clean up empty files and re-run with proper TTS"""
    
    # Paths
    base_dir = Path("./pipeline_output")
    video_dir = base_dir / f"video_{video_id}"
    audio_dir = video_dir / "translated_audio"
    
    print(f"Cleaning up and re-running translation for video {video_id}...")
    
    # Remove old audio directory if it exists
    if audio_dir.exists():
        print(f"Removing old audio directory: {audio_dir}")
        shutil.rmtree(audio_dir)
    
    # Load existing translation data
    translation_file = video_dir / "translation.json"
    if not translation_file.exists():
        print(f"Translation file not found: {translation_file}")
        return
    
    print("Loading existing translation data...")
    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_data = json.load(f)
    
    segments = translation_data.get('segments', [])
    print(f"Found {len(segments)} translation segments")
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        return
    
    # Initialize translator with proper TTS
    print("Initializing OpenAI TTS...")
    translator = AudioTranslator(target_language=target_language, tts_engine="openai")
    
    # Convert segments back to TranslationSegment objects
    from audio_translator import TranslationSegment
    translated_segments = []
    for seg in segments:
        translated_segments.append(TranslationSegment(
            start_time=seg['start_time'],
            end_time=seg['end_time'],
            original_text=seg['original_text'],
            translated_text=seg['translated_text']
        ))
    
    # Generate audio with proper TTS (limit to first 10 segments for testing)
    print("Generating audio with OpenAI TTS (first 10 segments for testing)...")
    test_segments = translated_segments[:10]  # Test with first 10 segments
    
    try:
        translated_audio_path = translator.generate_translated_audio(
            test_segments,
            str(audio_dir)
        )
        print(f"✅ Audio generated successfully: {translated_audio_path}")
        
        # Check if files were actually created
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("segment_*.wav"))
            print(f"Created {len(audio_files)} audio segment files")
            
            # Check if the files have content
            for i, audio_file in enumerate(audio_files[:3]):  # Check first 3
                size = audio_file.stat().st_size
                print(f"  {audio_file.name}: {size} bytes")
                
        else:
            print("❌ Audio directory was not created")
            
    except Exception as e:
        print(f"❌ Error generating audio: {str(e)}")
        return
    
    print("\n🎉 Test completed! If the audio files have content, you can generate the full audio.")
    print("To generate all segments, modify the script to use all translated_segments instead of test_segments[:10]")

if __name__ == "__main__":
    cleanup_and_rerun()