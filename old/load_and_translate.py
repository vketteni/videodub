#!/usr/bin/env python3
"""
Load existing scraped video and translate it to German
"""

import json
import os
from pathlib import Path
from audio_translator import AudioTranslator
from datetime import datetime

def load_and_translate_video(video_id="8XdL-7tAqnU", target_language="de"):
    """Load existing video data and translate it"""
    
    # Paths
    base_dir = Path("./pipeline_output")
    scraped_dir = base_dir / "scraped" / video_id
    metadata_file = scraped_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        return
    
    # Load metadata
    print("Loading video metadata...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Title: {metadata['metadata']['title']}")
    print(f"Duration: {metadata['metadata']['duration']} seconds")
    print(f"Transcript entries: {len(metadata.get('transcript', []))}")
    
    if not metadata.get('transcript'):
        print("No transcript found in metadata!")
        return
    
    # Create output directory for translation
    output_dir = base_dir / f"video_{video_id}"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize translator
    print(f"Initializing translator for {target_language}...")
    translator = AudioTranslator(target_language=target_language, tts_engine="openai")
    
    # Translate transcript
    print("Translating transcript...")
    transcript = metadata['transcript']
    translated_segments = translator.translate_transcript(transcript, target_language)
    
    print(f"Translated {len(translated_segments)} segments")
    
    # Save translation data
    translation_file = output_dir / "translation.json"
    translator.save_translation_data(translated_segments, str(translation_file))
    print(f"Translation saved to: {translation_file}")
    
    # Generate translated audio
    print("Generating translated audio...")
    audio_output_dir = output_dir / "translated_audio"
    try:
        translated_audio_path = translator.generate_translated_audio(
            translated_segments,
            str(audio_output_dir)
        )
        print(f"Translated audio saved to: {translated_audio_path}")
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        print("Audio generation failed, but translation text is saved.")
    
    # Create summary
    summary = {
        "video_id": video_id,
        "title": metadata['metadata']['title'],
        "original_language": "en",
        "target_language": target_language,
        "duration": metadata['metadata']['duration'],
        "transcript_segments": len(translated_segments),
        "processing_time": datetime.now().isoformat(),
        "files": {
            "original_audio": str(scraped_dir / f"{metadata['metadata']['title']}.mp3"),
            "translation_data": str(translation_file),
            "translated_audio": str(audio_output_dir / "translated_audio.wav")
        }
    }
    
    summary_file = output_dir / "translation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nTranslation completed!")
    print(f"Summary saved to: {summary_file}")
    print(f"Check the output directory: {output_dir}")

if __name__ == "__main__":
    load_and_translate_video()