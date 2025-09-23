# Image Sequence to MP4 Converter & Video Splitter

A powerful Python script that converts image sequences to MP4 videos and splits videos into frame-based chunks. Features advanced video processing including frame interpolation, slow motion effects, and intelligent video splitting with comprehensive debug information.

## Features

### 🎬 Image Sequence Conversion
- **Automatic Detection**: Finds image sequences by analyzing file patterns
- **Multiple Formats**: Supports JPEG, PNG, TIFF, TGA, BMP, EXR
- **Smart Naming**: Handles various naming conventions (e.g., `render_001.png`, `shot_v001_0001.exr`)
- **Batch Processing**: Converts all sequences in a directory at once

### 🎥 Video Processing
- **Frame Interpolation**: Upsample frame rates using motion-compensated interpolation
- **Slow Motion**: Create smooth slow-motion effects with timeline stretching
- **Custom Start Frames**: Override auto-detected frame numbers
- **Quality Control**: Adjustable codec, preset, CRF, and pixel format settings

### ✂️ Video Splitting
- **Frame-Based Chunks**: Split videos into exact frame counts (default: 81 frames)
- **Auto Frame Rate Detection**: Automatically detects video frame rates and total frame counts
- **Frame-Accurate Splitting**: Uses precise time-based cutting with proper timestamp handling
- **Real-Time Progress**: Live feedback showing chunk creation progress and frame verification
- **Batch Splitting**: Process multiple videos simultaneously
- **Debug Information**: Comprehensive output showing frame counts, durations, and file sizes

## Installation

### Prerequisites
- Python 3.6+
- FFmpeg (with ffprobe)

### FFmpeg Installation

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

## Usage

### Basic Image Sequence Conversion

```bash
# Convert all image sequences in current directory
python convert_sequences.py

# Specify output directory
python convert_sequences.py --output-dir videos

# Custom frame rate
python convert_sequences.py --frame-rate 24
```

### Advanced Video Processing

```bash
# Slow motion (3x slower)
python convert_sequences.py --frame-rate 48 --slow-motion 3.0

# Frame interpolation (upsample to 60fps)
python convert_sequences.py --interpolate-fps 60

# Custom start frame
python convert_sequences.py --start-frame 100

# High quality settings
python convert_sequences.py --crf 15 --preset slow --codec libx264
```

### Video Splitting

```bash
# Split all videos into 81-frame chunks
python convert_sequences.py --split-videos

# Custom chunk size
python convert_sequences.py --split-videos --chunk-frames 120

# Split videos from specific directory
python convert_sequences.py --split-videos --video-dir /path/to/videos

# Custom output directory
python convert_sequences.py --split-videos --output-dir chunks
```

## Command Line Options

### Image Sequence Conversion

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--frame-rate` | `-f` | 16 | Frame rate for output videos |
| `--output-dir` | `-o` | output | Output directory for MP4 files |
| `--codec` | `-c` | libx264 | Video codec |
| `--preset` | `-p` | medium | Encoding preset |
| `--crf` | | 18 | Quality setting (lower = better quality) |
| `--pixel-format` | | yuv420p | Pixel format |
| `--start-frame` | `-s` | auto | Custom start frame number |
| `--interpolate-fps` | `-i` | | Enable frame interpolation to specified FPS |
| `--interpolation-mode` | | mci | Interpolation mode (mci, blend, dup, me, mc) |
| `--slow-motion` | | | Create slow motion effect (factor) |
| `--dry-run` | | | Show what would be converted without converting |

### Video Splitting

| Option | Default | Description |
|--------|---------|-------------|
| `--split-videos` | | Enable video splitting mode |
| `--chunk-frames` | 81 | Number of frames per chunk |
| `--video-dir` | . | Directory to search for videos |
| `--output-dir` | output | Output directory for chunks |

## Examples

### Example 1: Basic Sequence Conversion
```bash
# Convert image sequences with default settings
python convert_sequences.py
```
**Input:** `render_0001.jpg`, `render_0002.jpg`, `render_0003.jpg`  
**Output:** `render.mp4` (16fps)

### Example 2: Slow Motion Effect
```bash
# Create 3x slow motion from 48fps footage
python convert_sequences.py --frame-rate 48 --slow-motion 3.0
```
**Input:** 48fps sequence (2 seconds)  
**Output:** 16fps video (6 seconds) with smooth slow motion

### Example 3: Frame Interpolation
```bash
# Upsample 24fps to 60fps with motion compensation
python convert_sequences.py --interpolate-fps 60 --interpolation-mode mci
```
**Input:** 24fps sequence  
**Output:** 60fps video with interpolated frames

### Example 4: Video Splitting
```bash
# Split videos into 81-frame chunks
python convert_sequences.py --split-videos --chunk-frames 81
```
**Input:** `video.mp4` (1000 frames at 24fps)  
**Output:** `video_chunk_000.mp4`, `video_chunk_001.mp4`, etc.

**Debug Output:**
```
🔄 Splitting: video
   Frame rate: 24.00fps
   Total frames: 1000
   Chunk size: 81 frames
   Will create: 13 chunks
   Video duration: 41.67s

   🔄 Creating chunk 1/13: frames 0-80 (81 frames)
   ✅ Created: video_chunk_000.mp4 (81 frames, 3.38s)
   🔄 Creating chunk 2/13: frames 81-161 (81 frames)
   ✅ Created: video_chunk_001.mp4 (81 frames, 3.38s)
   ...
```

## Supported File Formats

### Image Sequences
- **JPEG**: `.jpeg`, `.jpg`
- **PNG**: `.png`
- **TIFF**: `.tiff`
- **TGA**: `.tga`
- **BMP**: `.bmp`
- **EXR**: `.exr`

### Video Files
- **MP4**: `.mp4`
- **AVI**: `.avi`
- **MOV**: `.mov`
- **MKV**: `.mkv`
- **WMV**: `.wmv`
- **FLV**: `.flv`
- **WebM**: `.webm`
- **M4V**: `.m4v`

## Interpolation Modes

| Mode | Description | Speed | Quality |
|------|-------------|-------|---------|
| `mci` | Motion Compensated Interpolation | Slow | Best |
| `blend` | Blend frames together | Fast | Good |
| `dup` | Duplicate frames | Fastest | Fair |
| `me` | Motion Estimation only | Medium | Good |
| `mc` | Motion Compensation only | Medium | Good |

## Quality Settings

### CRF (Constant Rate Factor)
- **0**: Lossless (very large files)
- **18**: Visually lossless (recommended for high quality)
- **23**: Default (good quality)
- **28**: Lower quality (smaller files)

### Presets
- **ultrafast**: Fastest encoding, larger files
- **fast**: Fast encoding
- **medium**: Balanced (default)
- **slow**: Better compression
- **veryslow**: Best compression, slowest encoding

## Troubleshooting

### Common Issues

**"No image files found"**
- Check that your files follow the naming pattern: `name_0001.jpg`, `name_0002.jpg`
- Ensure files are in the current directory or specify the correct path

**"FFmpeg not found"**
- Install FFmpeg and ensure it's in your system PATH
- Test with: `ffmpeg -version`

**"Failed to convert"**
- Check file permissions
- Ensure output directory exists and is writable
- Verify FFmpeg can read your input files

**"Broken video chunks"**
- The script now uses frame-accurate splitting with proper timestamp handling
- If issues persist, try reducing chunk size or check input video integrity

**Slow processing**
- Use `--preset fast` for faster encoding
- Use `--crf 23` for smaller files
- Video splitting uses re-encoding for frame accuracy (slower but reliable)

### Debug Information

The script provides comprehensive debug output:
- **Real-time progress**: See which chunk is being created
- **Frame verification**: Each chunk is verified for correct frame count
- **File details**: Output shows file size and frame count for each chunk
- **Error details**: Detailed error messages for troubleshooting

### Performance Tips

1. **For large sequences**: Use `--preset fast` to speed up encoding
2. **For quality**: Use `--crf 15` and `--preset slow`
3. **For video splitting**: Uses re-encoding for frame accuracy (reliable but slower)
4. **For interpolation**: `mci` mode is slowest but highest quality
5. **Debug output**: Can be verbose for large batches - consider redirecting output to file

## License

This script is provided as-is for educational and personal use.

## Contributing

Feel free to submit issues and enhancement requests!
