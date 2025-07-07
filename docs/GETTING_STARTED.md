## Installation

```bash
# Install in development mode
pip install -e .

# OR install with development dependencies
make dev-install

# Set up environment variables (required for AI services)
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google/credentials.json"  # Optional
export AZURE_SPEECH_KEY="your_azure_key"  # Optional
```

### Development Setup
```bash
# Run all quality checks
make check

# Clean output files
make clean-output

# Run tests
make test
```

## Quick Start

### 🚀 **Fast Testing with Cost Analysis**

```bash
# Quick single video test with cost tracking
python examples/quick_test.py quick "https://youtube.com/watch?v=SHORT_VIDEO"

# Compare multiple models economically
python examples/quick_test.py compare

# Model comparison shows real costs:
# Model           Duration   Status     Total Cost    Tokens    
# gpt-3.5-turbo   45.2s      completed  $0.002340     1,247     
# gpt-4.1-nano    42.1s      completed  $0.000456     1,198     
# gpt-4           48.3s      completed  $0.024680     1,289     
```

### 📚 **Programmatic Usage**

```python
from pathlib import Path
from videodub import (
    TTSEngine, OpenAITranslationService, create_tts_service,
    create_audio_processing_service, FFmpegVideoProcessingService,
    FileStorageService
)
from videodub.core.pipeline import TranslationPipeline
from videodub.services.data_extraction import YouTubeDataExtractionService
from videodub.services.alignment import TimingAlignmentService
from videodub.core.models import PipelineConfig

# Create output directory
output_path = Path("./my_translations")
output_path.mkdir(parents=True, exist_ok=True)

# Create pipeline with modern architecture
pipeline = TranslationPipeline(
    data_extraction_service=YouTubeDataExtractionService(output_dir=output_path),
    translation_service=OpenAITranslationService(
        api_key="your_api_key",
        model="gpt-4.1-nano"  # Cost-effective model
    ),
    alignment_service=TimingAlignmentService(),
    tts_service=create_tts_service(engine=TTSEngine.OPENAI, openai_api_key="your_api_key"),
    audio_processing_service=create_audio_processing_service(),
    video_processing_service=FFmpegVideoProcessingService(),
    storage_service=FileStorageService(base_path=output_path),
    config=PipelineConfig(
        target_language="de",  # German
        tts_engine=TTSEngine.OPENAI,
        output_directory=str(output_path),
    )
)

# Process video with cost tracking
result = await pipeline.process_video("https://youtube.com/watch?v=VIDEO_ID")

# Check results and costs
print(f"Status: {result.status.value}")
print(f"Total Cost: ${result.cost_summary['totals']['total_cost']:.6f}")
print(f"Translation Strategy: {result.alignment_evaluation.best_strategy if result.alignment_evaluation else 'N/A'}")
```

### 🎯 **Advanced Features - Alignment Strategies**

```python
# Test multiple alignment strategies (A/B testing)
result = await pipeline.process_video_with_alignment_comparison("https://youtube.com/watch?v=VIDEO_ID")

# Check which alignment strategy worked best
if result.alignment_evaluation:
    print(f"Best Strategy: {result.alignment_evaluation.best_strategy}")
    print(f"Strategies Tested: {len(result.alignment_evaluation.strategy_results)}")
    
    for strategy, eval_result in result.alignment_evaluation.strategy_results.items():
        print(f"  {strategy}: Score {eval_result.overall_score:.3f} (Time: {eval_result.execution_time:.2f}s)")

# Sequential processing for multiple videos
urls = ["https://youtube.com/watch?v=VIDEO1", "https://youtube.com/watch?v=VIDEO2"]
for i, url in enumerate(urls):
    result = await pipeline.process_video(url)
    cost = result.cost_summary['totals']['total_cost'] if result.cost_summary else 0
    print(f"Video {i+1}: {result.status.value} - ${cost:.6f}")
```
