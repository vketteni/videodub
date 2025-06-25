# Video Scraper

Multi-platform video scraper for YouTube, Vimeo, and 1000+ other sites. Built with Python and powered by yt-dlp.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/vketteni/video-scraper.git
```

## Quick Start

### Simple Example

Create a new project and install the package:

```bash
mkdir my-video-project
cd my-video-project
pip install git+https://github.com/vketteni/video-scraper.git
```

Create `main.py`:

```python
from youtube_scraper import scrape_video

# Hardcoded example - download a video
result = scrape_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if result.success:
    print(f"✅ Successfully downloaded: {result.metadata.title}")
    print(f"📁 File saved to: {result.file_path}")
    print(f"⏱️  Duration: {result.metadata.duration} seconds")
    print(f"👁️  Views: {result.metadata.view_count}")
else:
    print(f"❌ Failed to download: {result.error}")
```

Run the example:

```bash
python main.py
```

### Advanced Usage

```python
from youtube_scraper import VideoScraperAPI

# Create scraper with custom settings
scraper = VideoScraperAPI(
    output_dir="./downloads",
    quality="720p",
    extract_audio=True
)

# Download video with metadata
result = scraper.scrape("https://vimeo.com/123456789")

if result.success:
    print(f"Title: {result.metadata.title}")
    print(f"Duration: {result.metadata.duration}s")
    print(f"File: {result.file_path}")
    
    # Access transcript if available
    if result.metadata.transcript:
        print("Transcript preview:")
        for entry in result.metadata.transcript[:3]:
            print(f"  {entry.start}s: {entry.text}")
```

### Audio-Only Download

```python
from youtube_scraper import scrape_audio_only

result = scrape_audio_only("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
print(f"Audio saved to: {result.file_path}")
```

### Metadata Only

```python
from youtube_scraper import scrape_metadata_only

result = scrape_metadata_only("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
print(f"Title: {result.metadata.title}")
print(f"Duration: {result.metadata.duration}s")
print(f"Views: {result.metadata.view_count}")
```

## Command Line Usage

After installation, use the command line tool:

```bash
# Download a video
video-scraper "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Download audio only
video-scraper --audio-only "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Custom output directory
video-scraper --output ./my-downloads "https://vimeo.com/123456789"
```

### Advanced Options

```bash
python main.py scrape "https://youtu.be/VIDEO_ID" \
  --output-dir ./my_output \
  --audio-format mp3 \
  --audio-quality 192 \
  --video-quality 1080p \
  --no-transcript
```

### Show Supported Platforms

```bash
python main.py platforms
```

### List Scraped Videos

```bash
python main.py list
```

### Export for Data Pipeline

```bash
python main.py export VIDEO_ID
```

### Delete Video Data

```bash
python main.py delete VIDEO_ID
```

## Output Structure

```
output/
├── {platform}_{video_id}/
│   ├── metadata.json          # Video metadata
│   ├── transcript.json        # Transcript with timestamps
│   ├── audio.mp3             # Audio file
│   ├── video.mp4             # Video file
│   └── pipeline_export.json  # Data pipeline format
```

## Supported Platforms

- **YouTube** - Full support with auto-generated transcripts
- **Vimeo** - Professional content with VTT subtitles
- **Dailymotion** - European video platform
- **Twitch** - Gaming and live content VODs
- **TikTok** - Short-form video content
- **Instagram** - Video posts and reels
- **Facebook** - Video posts and watch content
- **Twitter/X** - Video tweets
- **Reddit** - Video posts
- **And 1000+ more** via yt-dlp support

## Data Pipeline Integration

The `pipeline_export.json` file contains:

```json
{
  "video_id": "VIDEO_ID",
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "title": "Video Title",
  "channel": "Channel Name",
  "duration": 1234,
  "upload_date": "20231225",
  "scraped_at": "2024-01-01T12:00:00",
  "files": {
    "audio": "/path/to/audio.mp3",
    "video": "/path/to/video.mp4",
    "transcript": "/path/to/transcript.json",
    "metadata": "/path/to/metadata.json"
  },
  "transcript_entries": 150
}
```

## Configuration Options

- `--audio-format`: mp3, wav, m4a, etc.
- `--audio-quality`: Bitrate (128, 192, 320)
- `--video-format`: mp4, webm, mkv, etc.
- `--video-quality`: 360p, 720p, 1080p, etc.
- `--no-transcript`: Skip transcript extraction
- `--no-audio`: Skip audio extraction
- `--no-video`: Skip video extraction