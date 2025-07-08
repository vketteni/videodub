# VideoDub - AI Video Translation Pipeline

A comprehensive Python pipeline that downloads YouTube videos, translates their transcripts to different languages using advanced sentence reconstruction, generates natural-sounding audio in the target language with AI-powered text-to-speech services, and creates fully dubbed videos with real-time cost tracking.
## Supported Features

### 🌍 **Languages**
Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Russian, Arabic, Hindi

### 🤖 **Translation Models**
- **GPT-4.1-nano** - Latest, most cost-effective (recommended)
- **GPT-4-turbo** - High performance, balanced cost
- **GPT-4** - Premium quality, highest cost
- **GPT-3.5-turbo** - Fast, economical baseline

### 📹 **Video Sources**
YouTube and 1000+ platforms via yt-dlp integration

## Dependencies

The project relies on several key external services and packages:

- **video-scraper**: External package for video downloading
- **OpenAI API**: Translation and high-quality TTS
- **yt-dlp**: Underlying video extraction technology
- **ffmpeg**: Audio processing, manipulation, and video dubbing


### Core Components

#### 1. **Core Pipeline (`src/videodub/core/`)**
- **`pipeline.py`** - Pipeline orchestrator with alignment service integration
- **`models.py`** - Comprehensive data models with timing and alignment support
- **`interfaces.py`** - Service interfaces and contracts
- **`exceptions.py`** - Comprehensive error handling

#### 2. **Configuration System (`src/videodub/config/`)**
- **`settings.py`** - Pydantic-based configuration with validation
- **`validation.py`** - Input validation and sanitization
- Manages API keys for OpenAI, Google, and Azure services
- Defines supported languages and TTS engines with pricing

#### 3. **Data Extraction (`src/videodub/services/data_extraction.py`)**
- **`YouTubeDataExtractionService`** - Extracts transcript and timing metadata from YouTube videos
- Handles quality scoring and timing accuracy analysis
- Manages source type detection and validation
- Future support for speech-to-text for raw videos

#### 4. **Translation System (`src/videodub/services/translator.py`)**
- **Pure text-to-text translation** with clean separation of concerns
- **`OpenAITranslationService`** - Standard implementation for development
- **`OptimizedOpenAITranslationService`** - 90% performance improvement for production
- Support for multiple OpenAI models (GPT-3.5, GPT-4, GPT-4.1-nano)
- Batch processing and error handling with fallbacks

#### 5. **Alignment Service (`src/videodub/services/alignment.py`)**
- **`TimingAlignmentService`** - Synchronizes translated text with original timing
- Multiple alignment strategies: LENGTH_BASED, SENTENCE_BOUNDARY, HYBRID
- A/B testing support for strategy comparison and optimization
- Quality scoring: timing_accuracy, text_preservation, boundary_alignment

#### 6. **Text-to-Speech Engine (`src/videodub/services/tts.py`)**
- Multi-engine support (OpenAI, Google, Azure, System TTS)
- Sentence-based audio generation for natural speech flow
- Character usage tracking for cost analysis
- Quality optimization through complete sentence processing

#### 7. **Video Processing Engine (`src/videodub/services/video.py`)**
- FFmpeg-based video dubbing with audio replacement
- Lossless video quality preservation during audio track replacement
- Complete video metadata extraction (duration, resolution, codecs)
- Graceful fallback handling for video processing errors

#### 8. **Cost Tracking & Analytics (`src/videodub/utils/cost_tracking.py`)**
- Real-time API usage tracking
- Accurate cost calculation with current pricing models
- Token and character usage analytics
- Session-based cost summaries and breakdowns

### Utility Scripts & Examples

#### 8. **Testing & Development (`examples/`)**
##### TODO: MAKE CONCISE
- **`basic_usage.py`** - Comprehensive usage examples
  - Demonstrates different pipeline configurations
  - Shows single video and batch processing
  - Multi-language translation examples
- **`quick_test.py`** - Economic testing and model comparison
  - Fast single video testing with cost analysis
  - Multi-model comparison (GPT-3.5, GPT-4, GPT-4.1-nano)
  - Real-time cost tracking and performance metrics

#### 9. **Development Tools (`Makefile`)**
- **`make clean-output`** - Clean all pipeline outputs
- **`make clean-all`** - Complete cleanup (build + outputs)
- Standard development commands (test, lint, format, type-check)

## Pipeline Architecture

### Pipeline Workflow
#### TODO: CREATE
```mermaid
```

### Service Dependencies
#### TODO: CREATE
```mermaid
```

