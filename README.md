# YouTube Translation Pipeline

A comprehensive Python pipeline that downloads YouTube videos, translates their transcripts to different languages, and generates new audio in the target language using AI-powered text-to-speech services.

## Architecture Overview

This project consists of several interconnected modules that work together to create a complete video translation workflow:

```
YouTube Video → Audio Extraction → Transcript Translation → Audio Generation → Output
```

### Core Components

#### 1. **config.py** - Central Configuration
- Manages API keys for OpenAI, Google, and Azure services
- Defines supported languages and TTS engines
- Handles validation of required credentials
- Provides default settings for video processing

#### 2. **video_scraper.py** - Video Processing
- Wraps the external `youtube_scraper` package
- Downloads videos and extracts audio/metadata
- Handles transcript extraction when available
- Manages file organization and storage

#### 3. **audio_translator.py** - Translation & Speech Synthesis
- Translates transcripts using OpenAI GPT models
- Generates audio using multiple TTS engines (OpenAI, Google, Azure, System)
- Manages timing and synchronization of translated segments
- Combines individual audio segments into final output

#### 4. **translation_pipeline.py** - Main Orchestrator
- Coordinates the entire translation workflow
- Handles error management and logging
- Manages batch processing of multiple videos
- Provides status tracking and progress monitoring

### Utility Scripts

#### 5. **example_usage.py** - Usage Examples
- Demonstrates different pipeline configurations
- Shows single video and batch processing
- Provides examples for different TTS engines
- Includes status monitoring examples

#### 6. **load_and_translate.py** - Existing Video Processing
- Processes previously downloaded videos
- Useful for re-translation with different languages
- Demonstrates working with existing metadata

#### 7. **rerun_with_real_tts.py** - TTS Re-processing
- Cleans up failed audio generation attempts
- Re-runs audio generation with proper TTS engines
- Includes testing with limited segments

## Workflow Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YouTube URL   │───▶│  video_scraper  │───▶│   Raw Audio +   │
└─────────────────┘    │                 │    │   Transcript    │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Translated Audio│◀───│audio_translator │◀───│ Original Text   │
│     Output      │    │                 │    │   Segments      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## File Relationships

### Configuration Layer
- **config.py** ← Used by all modules for settings and API keys

### Processing Layer
- **translation_pipeline.py** ← Main entry point
  - Uses **video_scraper.py** for content extraction
  - Uses **audio_translator.py** for translation and TTS
  - Uses **config.py** for configuration

### Utility Layer
- **example_usage.py** ← Demonstrates **translation_pipeline.py**
- **load_and_translate.py** ← Direct usage of **audio_translator.py**
- **rerun_with_real_tts.py** ← Recovery and re-processing utilities

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional, for AI services)
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google/credentials.json"
export AZURE_SPEECH_KEY="your_azure_key"
```

## Quick Start

### Basic Usage

```python
from translation_pipeline import TranslationPipeline

# Initialize pipeline
pipeline = TranslationPipeline(
    target_language="es",  # Spanish
    tts_engine="openai"    # Requires API key
)

# Process a video
result = pipeline.process_video("https://www.youtube.com/watch?v=VIDEO_ID")
print(f"Status: {result['status']}")
```

### Command Line Usage

```bash
# Single video
python translation_pipeline.py "https://youtube.com/watch?v=VIDEO_ID" --target-language es

# Multiple videos
python translation_pipeline.py \
  "https://youtube.com/watch?v=VIDEO1" \
  "https://youtube.com/watch?v=VIDEO2" \
  --target-language fr \
  --tts-engine openai

# Check pipeline status
python translation_pipeline.py --status
```

## Supported Features

- **Languages**: Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Russian, Arabic, Hindi
- **TTS Engines**: OpenAI (premium), Google Cloud, Azure Speech, System TTS (fallback)
- **Video Sources**: YouTube and 1000+ other platforms via yt-dlp
- **Output Formats**: MP3 audio, JSON metadata, synchronized transcripts

## Output Structure

```
pipeline_output/
├── video_{VIDEO_ID}/
│   ├── metadata.json           # Video information
│   ├── translation.json        # Translated segments
│   ├── translated_audio/       # Generated audio files
│   │   ├── segment_0001.wav
│   │   ├── segment_0002.wav
│   │   └── translated_audio.wav # Combined final audio
│   └── pipeline_result.json    # Processing summary
```

## Dependencies

The project relies on several key external services and packages:

- **youtube_scraper**: External package for video downloading
- **OpenAI API**: Translation and high-quality TTS
- **yt-dlp**: Underlying video extraction technology
- **ffmpeg**: Audio processing and manipulation

## Error Handling

The pipeline includes comprehensive error handling:
- Graceful fallbacks for missing API keys
- Retry mechanisms for network issues
- Detailed logging for debugging
- Partial processing recovery