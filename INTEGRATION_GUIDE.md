# VideoDub - Integration Guide

This guide shows how to integrate VideoDub (AI Video Translation Pipeline) as a package in your own projects.

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/vketteni/videodub.git
cd videodub

# Install in development mode
pip install -e .

# Or install with development dependencies
make dev-install
```

### Option 2: Install as Dependency

Add to your `requirements.txt`:
```
git+https://github.com/vketteni/videodub.git
```

Or install directly:
```bash
pip install git+https://github.com/vketteni/videodub.git
```

## Prerequisites

### System Dependencies

Ensure you have FFmpeg installed:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

### API Keys

Set up your environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional: For additional TTS engines
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google/credentials.json"
export AZURE_SPEECH_KEY="your_azure_speech_key"
```

## Quick Start

### Basic Usage

```python
import asyncio
from videodub import create_pipeline, TTSEngine

async def translate_video():
    # Create pipeline with default settings
    pipeline = create_pipeline(
        output_directory="./my_translations",
        target_language="es",  # Spanish
        tts_engine=TTSEngine.OPENAI,
        openai_api_key="your_api_key"
    )
    
    # Process a video
    result = await pipeline.process_video("https://youtube.com/watch?v=VIDEO_ID")
    
    # Check results
    if result.status.value == "completed":
        print(f"✅ Translation completed!")
        print(f"📁 Files: {result.files}")
        print(f"💰 Total cost: ${result.cost_summary['totals']['total_cost']:.6f}")
    else:
        print(f"❌ Translation failed: {result.errors}")

# Run the translation
asyncio.run(translate_video())
```

### Advanced Configuration

```python
from videodub import create_pipeline, TTSEngine
from pathlib import Path

async def advanced_translation():
    pipeline = create_pipeline(
        output_directory="./advanced_output",
        target_language="de",           # German
        translation_model="gpt-4.1-nano",  # Cost-effective model
        tts_model="tts-1-hd",          # High-definition audio
        tts_engine=TTSEngine.OPENAI,
        max_concurrent_requests=3,      # Slower but more reliable
        openai_api_key="your_api_key"
    )
    
    url = "https://youtube.com/watch?v=VIDEO_ID"
    result = await pipeline.process_video(url)
    
    return result

result = asyncio.run(advanced_translation())
```

## Integration Patterns

### 1. Flask Web Application

```python
from flask import Flask, request, jsonify
from videodub import create_pipeline, TTSEngine
import asyncio
import threading

app = Flask(__name__)

# Initialize pipeline once
pipeline = create_pipeline(
    output_directory="./web_translations",
    target_language="es",
    tts_engine=TTSEngine.OPENAI,
    openai_api_key="your_api_key"
)

def run_translation(url, result_dict):
    """Run translation in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(pipeline.process_video(url))
        result_dict['result'] = result
    finally:
        loop.close()

@app.route('/translate', methods=['POST'])
def translate_video():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    # Run translation in background thread
    result_dict = {}
    thread = threading.Thread(target=run_translation, args=(url, result_dict))
    thread.start()
    thread.join(timeout=300)  # 5 minute timeout
    
    if thread.is_alive():
        return jsonify({'error': 'Translation timeout'}), 408
    
    result = result_dict.get('result')
    if result and result.status.value == "completed":
        return jsonify({
            'status': 'success',
            'files': result.files,
            'cost': result.cost_summary['totals']['total_cost']
        })
    else:
        return jsonify({
            'status': 'failed',
            'errors': result.errors if result else ['Unknown error']
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. FastAPI Async Application

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from videodub import create_pipeline, TTSEngine
import asyncio
from typing import Dict
import uuid

app = FastAPI(title="YouTube Translation API")

# Initialize pipeline
pipeline = create_pipeline(
    output_directory="./api_translations",
    target_language="fr",  # French
    tts_engine=TTSEngine.OPENAI,
    openai_api_key="your_api_key"
)

# Store translation jobs
translation_jobs: Dict[str, dict] = {}

class TranslationRequest(BaseModel):
    url: str
    target_language: str = "es"

class TranslationResponse(BaseModel):
    job_id: str
    status: str
    message: str

async def process_translation(job_id: str, url: str, target_language: str):
    """Background task for processing translation."""
    try:
        translation_jobs[job_id]["status"] = "processing"
        
        # Create pipeline with specific language
        job_pipeline = create_pipeline(
            output_directory=f"./api_translations/{job_id}",
            target_language=target_language,
            tts_engine=TTSEngine.OPENAI,
            openai_api_key="your_api_key"
        )
        
        result = await job_pipeline.process_video(url)
        
        translation_jobs[job_id].update({
            "status": "completed" if result.status.value == "completed" else "failed",
            "result": {
                "files": result.files,
                "cost": result.cost_summary['totals']['total_cost'] if result.cost_summary else 0,
                "errors": result.errors
            }
        })
        
    except Exception as e:
        translation_jobs[job_id].update({
            "status": "failed",
            "result": {"errors": [str(e)]}
        })

@app.post("/translate", response_model=TranslationResponse)
async def translate_video(request: TranslationRequest, background_tasks: BackgroundTasks):
    """Start a video translation job."""
    job_id = str(uuid.uuid4())
    
    translation_jobs[job_id] = {
        "status": "queued",
        "url": request.url,
        "target_language": request.target_language
    }
    
    background_tasks.add_task(
        process_translation, 
        job_id, 
        request.url, 
        request.target_language
    )
    
    return TranslationResponse(
        job_id=job_id,
        status="queued",
        message="Translation job started"
    )

@app.get("/translate/{job_id}")
async def get_translation_status(job_id: str):
    """Get the status of a translation job."""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return translation_jobs[job_id]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipeline": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. CLI Tool Integration

```python
#!/usr/bin/env python3
"""Custom CLI tool using the translation pipeline."""

import asyncio
import click
from pathlib import Path
from videodub import create_pipeline, TTSEngine

@click.group()
def cli():
    """Custom YouTube Translation CLI."""
    pass

@cli.command()
@click.argument('url')
@click.option('--language', '-l', default='es', help='Target language (default: es)')
@click.option('--output', '-o', default='./translations', help='Output directory')
@click.option('--model', default='gpt-4.1-nano', help='Translation model')
@click.option('--hd-audio', is_flag=True, help='Use high-definition TTS')
@click.option('--api-key', envvar='OPENAI_API_KEY', required=True, help='OpenAI API key')
def translate(url, language, output, model, hd_audio, api_key):
    """Translate a YouTube video."""
    
    async def run_translation():
        click.echo(f"🎬 Translating video to {language}...")
        
        pipeline = create_pipeline(
            output_directory=output,
            target_language=language,
            translation_model=model,
            tts_model="tts-1-hd" if hd_audio else "tts-1",
            tts_engine=TTSEngine.OPENAI,
            openai_api_key=api_key
        )
        
        result = await pipeline.process_video(url)
        
        if result.status.value == "completed":
            click.echo("✅ Translation completed!")
            click.echo(f"📁 Output directory: {result.files.get('audio_directory', 'N/A')}")
            
            if 'dubbed_video' in result.files:
                click.echo(f"🎥 Dubbed video: {result.files['dubbed_video']}")
            
            if result.cost_summary:
                cost = result.cost_summary['totals']['total_cost']
                click.echo(f"💰 Total cost: ${cost:.6f}")
        else:
            click.echo("❌ Translation failed!")
            for error in result.errors:
                click.echo(f"   Error: {error}")
    
    asyncio.run(run_translation())

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def analyze(directory):
    """Analyze translation output directory."""
    output_dir = Path(directory)
    
    video_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('video_')]
    
    click.echo(f"📊 Analysis of {output_dir}")
    click.echo(f"   Found {len(video_dirs)} video translations")
    
    total_cost = 0
    for video_dir in video_dirs:
        result_file = video_dir / "pipeline_result.json"
        if result_file.exists():
            import json
            with open(result_file) as f:
                data = json.load(f)
                if 'cost_summary' in data and data['cost_summary']:
                    cost = data['cost_summary']['totals']['total_cost']
                    total_cost += cost
                    click.echo(f"   📹 {video_dir.name}: ${cost:.6f}")
    
    click.echo(f"💰 Total estimated cost: ${total_cost:.6f}")

if __name__ == '__main__':
    cli()
```

### 4. Batch Processing

```python
import asyncio
from videodub import create_pipeline, TTSEngine
from pathlib import Path
import json

async def batch_translate_videos():
    """Process multiple videos efficiently."""
    
    # List of videos to translate
    video_urls = [
        "https://youtube.com/watch?v=VIDEO1",
        "https://youtube.com/watch?v=VIDEO2",
        "https://youtube.com/watch?v=VIDEO3",
    ]
    
    # Target languages
    languages = ["es", "fr", "de"]
    
    total_cost = 0
    
    for language in languages:
        print(f"\n🌍 Processing language: {language}")
        
        pipeline = create_pipeline(
            output_directory=f"./batch_output/{language}",
            target_language=language,
            translation_model="gpt-4.1-nano",  # Cost-effective
            tts_engine=TTSEngine.OPENAI,
            openai_api_key="your_api_key"
        )
        
        # Process videos sequentially to avoid rate limits
        for i, url in enumerate(video_urls, 1):
            print(f"   📹 Processing video {i}/{len(video_urls)}")
            
            try:
                result = await pipeline.process_video(url)
                
                if result.status.value == "completed":
                    cost = result.cost_summary['totals']['total_cost']
                    total_cost += cost
                    print(f"   ✅ Completed - Cost: ${cost:.6f}")
                    
                    # Log successful translation
                    if 'dubbed_video' in result.files:
                        print(f"   🎥 Dubbed video: {result.files['dubbed_video']}")
                else:
                    print(f"   ❌ Failed: {result.errors}")
                    
            except Exception as e:
                print(f"   💥 Error: {e}")
            
            # Small delay between videos
            await asyncio.sleep(2)
    
    print(f"\n💰 Total batch cost: ${total_cost:.6f}")

if __name__ == "__main__":
    asyncio.run(batch_translate_videos())
```

## Configuration Options

### Pipeline Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_directory` | `str` | `"./output"` | Directory for output files |
| `target_language` | `str` | `"es"` | Target language code |
| `tts_engine` | `TTSEngine` | `TTSEngine.OPENAI` | TTS engine to use |
| `translation_model` | `str` | `"gpt-4.1-nano"` | OpenAI translation model |
| `tts_model` | `str` | `"tts-1"` | OpenAI TTS model |
| `max_concurrent_requests` | `int` | `5` | Max concurrent API requests |

### Supported Languages

- `es` - Spanish
- `fr` - French  
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi

### TTS Models

- `tts-1` - Standard quality, faster, cheaper
- `tts-1-hd` - High definition, slower, more expensive

## Output Structure

When you run the pipeline, it creates this output structure:

```
your_output_directory/
├── scraped/                    # Original downloaded videos
│   └── {video_id}/
│       └── video.mp4
├── video_{video_id}/           # Translation results
│   ├── metadata.json          # Video information
│   ├── translation.json       # Translated segments
│   ├── translated_audio/      # Audio files
│   │   ├── segment_0001.wav
│   │   ├── segment_0002.wav
│   │   └── translated_audio.wav
│   ├── dubbed_video_{lang}.mp4 # Final dubbed video
│   └── pipeline_result.json   # Processing summary
```

## Error Handling

```python
from videodub import create_pipeline, PipelineError, TTSError

async def robust_translation():
    pipeline = create_pipeline(
        target_language="es",
        openai_api_key="your_api_key"
    )
    
    try:
        result = await pipeline.process_video("https://youtube.com/watch?v=VIDEO_ID")
        
        if result.status.value == "completed":
            return result.files
        else:
            print(f"Translation failed: {result.errors}")
            return None
            
    except PipelineError as e:
        print(f"Pipeline error: {e}")
    except TTSError as e:
        print(f"TTS error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None
```

## Performance Tips

1. **Use `gpt-4.1-nano`** for cost-effective translations
2. **Set `max_concurrent_requests=2`** for rate limit compliance
3. **Use `tts-1`** instead of `tts-1-hd` for faster processing
4. **Process videos sequentially** for large batches
5. **Monitor costs** with `result.cost_summary`

## Monitoring and Logging

```python
import logging
from videodub import setup_pipeline_logging

# Configure logging
setup_pipeline_logging(level=logging.INFO)

# Your translation code here
```

## Docker Integration

```dockerfile
FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install pipeline
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install your app
COPY . /app
WORKDIR /app

CMD ["python", "your_app.py"]
```

## Support

For issues and questions:
- 📖 Check the main [README.md](README.md)
- 🐛 Report bugs on [GitHub Issues](https://github.com/vketteni/videodub/issues)
- 📧 Contact: [your-email@example.com]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.