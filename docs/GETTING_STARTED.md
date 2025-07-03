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
from videodub import create_pipeline, TTSEngine

# Create pipeline with sentence reconstruction (automatic)
pipeline = create_pipeline(
    target_language="de",           # German
    tts_engine=TTSEngine.OPENAI,   # High-quality TTS
    translation_model="gpt-4.1-nano",  # Cost-effective model
    openai_api_key="your_api_key"
)

# Process video with cost tracking
result = await pipeline.process_video("https://youtube.com/watch?v=VIDEO_ID")

# Check results and costs
print(f"Status: {result.status.value}")
print(f"Total Cost: ${result.cost_summary['totals']['total_cost']:.6f}")
print(f"Translation Segments: {len(result.translation_segments)}")
```

### 🎯 **Advanced Features**

```python
# Comprehensive example with all features
pipeline = create_pipeline(
    output_directory="./my_translations",
    target_language="de",           # German
    translation_model="gpt-4.1-nano",  # Latest efficient model
    tts_model="tts-1-hd",          # High-definition audio
    tts_engine=TTSEngine.OPENAI,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Batch processing with cost optimization
urls = ["https://youtube.com/watch?v=VIDEO1", "https://youtube.com/watch?v=VIDEO2"]
async for result in pipeline.process_video_batch(urls, max_concurrent=2):
    cost = result.cost_summary['totals']['total_cost']
    print(f"Video {result.video_id}: ${cost:.6f}")
```
