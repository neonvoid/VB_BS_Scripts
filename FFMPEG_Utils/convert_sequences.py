
"""
Image Sequence to MP4 Converter
Automatically detects and converts all image sequences in the current directory to MP4 videos.
"""

import os
import re
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def find_videos(directory="."):
    """
    Scan directory for video files and return a list of video file paths.
    """
    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f"*{ext}"))
    
    if not video_files:
        print("No video files found in directory!")
        return []
    
    print(f"Found {len(video_files)} video files")
    return [str(f) for f in video_files]

def split_video_into_chunks(video_path, output_dir, chunk_frames=81, frame_rate=None):
    """
    Split a video into chunks of specified frame count.
    """
    video_name = Path(video_path).stem
    safe_name = re.sub(r'[^\w\-_.]', '_', video_name)
    
    # Get video info to determine frame rate and total frames
    if frame_rate is None:
        try:
            # Get video frame rate using ffprobe
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            frame_rate_str = result.stdout.strip()
            if '/' in frame_rate_str:
                num, den = frame_rate_str.split('/')
                frame_rate = float(num) / float(den)
            else:
                frame_rate = float(frame_rate_str)
        except:
            frame_rate = 24.0  # Default fallback
            print(f"   Warning: Could not detect frame rate for {video_name}, using {frame_rate}fps")
    
    # Get total number of frames
    try:
        frame_count_cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(frame_count_cmd, capture_output=True, text=True, check=True)
        total_frames = int(result.stdout.strip())
    except:
        print(f"   Warning: Could not detect frame count for {video_name}")
        return False, 0
    
    print(f"   Splitting: {video_name}")
    print(f"   Frame rate: {frame_rate:.2f}fps")
    print(f"   Total frames: {total_frames}")
    print(f"   Chunk size: {chunk_frames} frames")
    
    # Calculate number of chunks needed
    num_chunks = (total_frames + chunk_frames - 1) // chunk_frames  # Ceiling division
    print(f"   Will create: {num_chunks} chunks")
    print(f"   Video duration: {total_frames/frame_rate:.2f}s")
    print()
    
    # Split video into frame-based chunks using time-based approach
    chunk_files = []
    
    for i in range(num_chunks):
        start_frame = i * chunk_frames
        end_frame = min((i + 1) * chunk_frames - 1, total_frames - 1)
        actual_frames = end_frame - start_frame + 1
        
        # Calculate precise start time and duration
        start_time = start_frame / frame_rate
        duration = actual_frames / frame_rate
        
        chunk_filename = f"{safe_name}_chunk_{i:03d}.mp4"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Use precise time-based splitting with proper seeking
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-ss', str(start_time),  # Start time
            '-i', video_path,  # Input after seeking
            '-t', str(duration),  # Duration
            '-c:v', 'libx264',  # Re-encode video
            '-c:a', 'aac',  # Re-encode audio
            '-preset', 'fast',  # Fast encoding
            '-crf', '18',  # Good quality
            '-avoid_negative_ts', 'make_zero',  # Fix timestamps
            '-fflags', '+genpts',  # Generate presentation timestamps
            chunk_path
        ]
        
        try:
            print(f"   Creating chunk {i+1}/{num_chunks}: frames {start_frame}-{end_frame} ({actual_frames} frames)")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            chunk_files.append(chunk_filename)
            
            # Get actual chunk info for verification
            try:
                chunk_info_cmd = [
                    'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                    '-count_frames', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', chunk_path
                ]
                chunk_result = subprocess.run(chunk_info_cmd, capture_output=True, text=True, check=True)
                actual_chunk_frames = int(chunk_result.stdout.strip())
                chunk_duration = actual_chunk_frames / frame_rate
                print(f"    Created: {chunk_filename} ({actual_chunk_frames} frames, {chunk_duration:.2f}s)")
            except:
                print(f"    Created: {chunk_filename} ({actual_frames} frames)")
                
        except subprocess.CalledProcessError as e:
            print(f"   Failed to create chunk {i+1}: {e}")
            if e.stderr:
                print(f"      Error: {e.stderr}")
    
    print(f"Successfully split into {len(chunk_files)} chunks")
    return True, len(chunk_files)

def process_single_file(file_path, output_dir, frame_rate=16, 
                       codec='libx264', preset='medium', crf=18, pixel_format='yuv420p', 
                       start_frame=None, interpolate_fps=None, interpolation_mode='mci', 
                       slow_motion_factor=None, reverse_slow_motion_factor=None):
    """
    Process a single file (image sequence or video) and convert it to MP4.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Check if it's a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    if file_path.suffix.lower() in video_extensions:
        print(f"Processing video file: {file_path.name}")
        # For video files, we'll just copy/convert them with the specified settings
        output_file = os.path.join(output_dir, f"{file_path.stem}.mp4")
        
        cmd = ['ffmpeg', '-y', '-i', str(file_path)]
        
        # Add video filters
        if slow_motion_factor is not None:
            target_fps = frame_rate / slow_motion_factor
            filter_complex = f"setpts={slow_motion_factor}*PTS,fps={target_fps}"
            cmd.extend(['-vf', filter_complex])
        elif reverse_slow_motion_factor is not None:
            filter_complex = f"setpts={1.0/reverse_slow_motion_factor}*PTS"
            cmd.extend(['-vf', filter_complex])
        elif interpolate_fps is not None:
            if interpolate_fps > frame_rate:
                filter_complex = f"minterpolate=fps={interpolate_fps}:mi_mode={interpolation_mode}"
            else:
                filter_complex = f"fps={interpolate_fps}"
            cmd.extend(['-vf', filter_complex])
        else:
            # Just set the frame rate
            cmd.extend(['-r', str(frame_rate)])
        
        cmd.extend([
            '-c:v', codec,
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', pixel_format,
            output_file
        ])
        
        print(f"   Converting video: {file_path.name}")
        if slow_motion_factor is not None:
            target_fps = frame_rate / slow_motion_factor
            print(f"   Slow Motion: {frame_rate}fps → {target_fps:.1f}fps ({slow_motion_factor}x slower)")
        elif reverse_slow_motion_factor is not None:
            print(f"   Reverse Slow Motion: {reverse_slow_motion_factor}x speed up (maintaining {frame_rate}fps)")
        elif interpolate_fps is not None:
            if interpolate_fps > frame_rate:
                print(f"   Upsampling: {frame_rate}fps → {interpolate_fps}fps ({interpolation_mode} mode)")
            else:
                print(f"   Downsampling: {frame_rate}fps → {interpolate_fps}fps (frame dropping)")
        else:
            print(f"   Frame Rate: {frame_rate}fps")
        print(f"   Output: {output_file}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully converted: {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {file_path.name}: {e}")
            if e.stderr:
                print(f"   Error: {e.stderr}")
            return False
    
    # Check if it's an image file
    image_extensions = {'.jpeg', '.jpg', '.png', '.tiff', '.tga', '.bmp', '.exr'}
    if file_path.suffix.lower() in image_extensions:
        print(f"Processing single image: {file_path.name}")
        # For single images, create a 1-frame video
        output_file = os.path.join(output_dir, f"{file_path.stem}.mp4")
        
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',  # Loop the image
            '-i', str(file_path),
            '-t', '1',  # 1 second duration
            '-r', str(frame_rate),
            '-c:v', codec,
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', pixel_format,
            output_file
        ]
        
        print(f"   Converting single image: {file_path.name}")
        print(f"   Duration: 1 second at {frame_rate}fps")
        print(f"   Output: {output_file}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully converted: {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {file_path.name}: {e}")
            if e.stderr:
                print(f"   Error: {e.stderr}")
            return False
    
    print(f"Unsupported file type: {file_path.suffix}")
    return False

def find_image_sequences(directory="."):
    """
    Scan directory for image sequences and group them by pattern.
    Returns a dictionary of {sequence_name: [frame_numbers]}
    """
    # Supported image extensions
    image_extensions = {'.jpeg', '.jpg', '.png', '.tiff', '.tga', '.bmp', '.exr'}
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
    
    if not image_files:
        print("No image files found in directory!")
        return {}
    
    print(f"Found {len(image_files)} image files")
    
    # Group files by sequence pattern
    sequences = defaultdict(list)
    
    for file_path in image_files:
        filename = file_path.name
        
        # Try to extract sequence name and frame number
        # Pattern: {prefix}.{frame_number}.{extension}
        # This handles various naming patterns like:
        # - Seq_KnightASharpeningSword.MP.beauty.0360.jpeg
        # - render_001.png
        # - shot_v001_0001.exr
        
        # Look for frame numbers at the end of the filename
        frame_match = re.search(r'(\d{3,})\.(jpeg|jpg|png|tiff|tga|bmp|exr)$', filename)
        if frame_match:
            frame_num = frame_match.group(1)
            # Extract the sequence name (everything before the frame number)
            sequence_name = filename[:frame_match.start()]
            sequences[sequence_name].append(int(frame_num))
            continue
        
        # Alternative pattern: frame number with underscore or dot
        frame_match = re.search(r'[._](\d{3,})\.(jpeg|jpg|png|tiff|tga|bmp|exr)$', filename)
        if frame_match:
            frame_num = frame_match.group(1)
            sequence_name = filename[:frame_match.start()]
            sequences[sequence_name].append(int(frame_num))
            continue
    
    # Filter out sequences with only one frame
    valid_sequences = {name: frames for name, frames in sequences.items() if len(frames) > 1}
    
    return valid_sequences

def convert_sequence_to_mp4(sequence_name, frame_numbers, output_dir, frame_rate=16, 
                           codec='libx264', preset='medium', crf=18, pixel_format='yuv420p', start_frame=None, 
                           interpolate_fps=None, interpolation_mode='mci', slow_motion_factor=None, 
                           reverse_slow_motion_factor=None):
    """
    Convert an image sequence to MP4 using ffmpeg.
    """
    # Determine the file extension from the first file
    first_frame = min(frame_numbers)
    # Find a file with this frame number to get the extension
    for ext in ['.jpeg', '.jpg', '.png', '.tiff', '.tga', '.bmp', '.exr']:
        test_file = f"{sequence_name}{first_frame:04d}{ext}"
        if os.path.exists(test_file):
            file_ext = ext
            break
    else:
        print(f"Could not determine file extension for sequence: {sequence_name}")
        return False
    
    # Create input pattern
    input_pattern = f"{sequence_name}%04d{file_ext}"
    
    # Create output filename
    safe_name = re.sub(r'[^\w\-_.]', '_', sequence_name.rstrip('._'))
    output_file = os.path.join(output_dir, f"{safe_name}.mp4")
    
    # Determine start frame
    actual_start_frame = start_frame if start_frame is not None else min(frame_numbers)
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-framerate', str(frame_rate),
        '-start_number', str(actual_start_frame),  # Start from the specified or first frame number
        '-i', input_pattern,
    ]
    
    # Add video filter for interpolation, slow motion, or reverse slow motion
    if slow_motion_factor is not None:
        # Slow motion: stretch timeline and reduce frame rate
        target_fps = frame_rate / slow_motion_factor
        filter_complex = f"setpts={slow_motion_factor}*PTS,fps={target_fps}"
        cmd.extend(['-vf', filter_complex])
    elif reverse_slow_motion_factor is not None:
        # Reverse slow motion: compress timeline and maintain frame rate
        filter_complex = f"setpts={1.0/reverse_slow_motion_factor}*PTS"
        cmd.extend(['-vf', filter_complex])
    elif interpolate_fps is not None:
        if interpolate_fps > frame_rate:
            # Upsampling: use minterpolate for motion-compensated interpolation
            filter_complex = f"minterpolate=fps={interpolate_fps}:mi_mode={interpolation_mode}"
        else:
            # Downsampling: use fps filter to drop frames
            filter_complex = f"fps={interpolate_fps}"
        cmd.extend(['-vf', filter_complex])
    
    # Add encoding options
    cmd.extend([
        '-c:v', codec,
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', pixel_format,
        output_file
    ])
    
    print(f"Converting: {sequence_name}")
    print(f"   Frames: {min(frame_numbers)}-{max(frame_numbers)} ({len(frame_numbers)} total)")
    if slow_motion_factor is not None:
        target_fps = frame_rate / slow_motion_factor
        print(f"   Slow Motion: {frame_rate}fps → {target_fps:.1f}fps ({slow_motion_factor}x slower)")
    elif reverse_slow_motion_factor is not None:
        print(f"   Reverse Slow Motion: {reverse_slow_motion_factor}x speed up (maintaining {frame_rate}fps)")
    elif interpolate_fps is not None:
        if interpolate_fps > frame_rate:
            print(f"   Upsampling: {frame_rate}fps → {interpolate_fps}fps ({interpolation_mode} mode)")
        else:
            print(f"   Downsampling: {frame_rate}fps → {interpolate_fps}fps (frame dropping)")
    print(f"   Output: {output_file}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {sequence_name}: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert image sequences to MP4 videos')
    parser.add_argument('files', nargs='*', 
                       help='Specific files to process (image sequences or videos). If not provided, auto-detects all sequences/videos in directory.')
    parser.add_argument('--frame-rate', '-f', type=int, default=16, 
                       help='Frame rate for output videos (default: 16)')
    parser.add_argument('--output-dir', '-o', default='output', 
                       help='Output directory for MP4 files (default: output)')
    parser.add_argument('--codec', '-c', default='libx264', 
                       help='Video codec (default: libx264)')
    parser.add_argument('--preset', '-p', default='medium', 
                       help='Encoding preset (default: medium)')
    parser.add_argument('--crf', type=int, default=18, 
                       help='Quality setting (default: 18)')
    parser.add_argument('--pixel-format', default='yuv420p', 
                       help='Pixel format (default: yuv420p)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be converted without actually converting')
    parser.add_argument('--start-frame', '-s', type=int, 
                       help='Custom start frame number for all sequences (overrides auto-detection)')
    parser.add_argument('--interpolate-fps', '-i', type=int, 
                       help='Enable frame interpolation to specified FPS using minterpolate filter')
    parser.add_argument('--interpolation-mode', default='mci', 
                       choices=['mci', 'blend', 'dup', 'me', 'mc'],
                       help='Interpolation mode: mci=motion compensated, blend=blend frames, dup=duplicate, me=motion estimation, mc=motion compensation (default: mci)')
    parser.add_argument('--slow-motion', type=float, 
                       help='Create slow motion effect: factor of how much slower (e.g., 3.0 for 3x slower)')
    parser.add_argument('--reverse-slow-motion', type=float, 
                       help='Reverse slow motion effect: factor of how much faster (e.g., 3.0 for 3x speed up)')
    parser.add_argument('--split-videos', action='store_true',
                       help='Split all videos in directory into 81-frame chunks')
    parser.add_argument('--chunk-frames', type=int, default=81,
                       help='Number of frames per chunk when splitting videos (default: 81)')
    parser.add_argument('--video-dir', default='.',
                       help='Directory to search for videos (default: current directory)')
    
    args = parser.parse_args()
    
    if args.split_videos:
        print("Video Splitter")
        print("=" * 50)
        print(f"Video Directory: {args.video_dir}")
        print(f"Chunk Size: {args.chunk_frames} frames")
        print(f"Output Directory: {args.output_dir}")
        print()
    else:
        print("Image Sequence to MP4 Converter")
        print("=" * 50)
        print(f"Frame Rate: {args.frame_rate} fps")
        print(f"Output Directory: {args.output_dir}")
        print(f"Codec: {args.codec}")
        print(f"Preset: {args.preset}")
        print(f"CRF: {args.crf}")
        if args.start_frame is not None:
            print(f"Start Frame: {args.start_frame}")
        if args.slow_motion is not None:
            target_fps = args.frame_rate / args.slow_motion
            print(f"Slow Motion: {args.frame_rate}fps → {target_fps:.1f}fps ({args.slow_motion}x slower)")
        elif args.reverse_slow_motion is not None:
            print(f"Reverse Slow Motion: {args.reverse_slow_motion}x speed up (maintaining {args.frame_rate}fps)")
        elif args.interpolate_fps is not None:
            if args.interpolate_fps > args.frame_rate:
                print(f"Frame Upsampling: {args.frame_rate}fps → {args.interpolate_fps}fps ({args.interpolation_mode} mode)")
            else:
                print(f"Frame Downsampling: {args.frame_rate}fps → {args.interpolate_fps}fps (frame dropping)")
        print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.split_videos:
        # Video splitting mode
        videos = find_videos(args.video_dir)
        
        if not videos:
            print("No video files found!")
            return 1
        
        print(f"Found {len(videos)} video files:")
        for video in videos:
            print(f"   • {os.path.basename(video)}")
        print()
        
        if args.dry_run:
            print("Dry run mode - no files will be split")
            return 0
        
        # Split each video
        successful = 0
        failed = 0
        total_chunks = 0
        
        for video_path in videos:
            success, chunk_count = split_video_into_chunks(
                video_path, args.output_dir, args.chunk_frames
            )
            if success:
                successful += 1
                total_chunks += chunk_count
            else:
                failed += 1
            print()
        
        # Summary
        print("=" * 50)
        print(f"Video splitting complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total chunks created: {total_chunks}")
        
        if successful > 0:
            print(f"\n Output files in '{args.output_dir}' directory:")
            chunk_files = [f for f in os.listdir(args.output_dir) if f.endswith('.mp4') and '_chunk_' in f]
            for file in sorted(chunk_files):
                file_path = os.path.join(args.output_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                # Get frame count for each chunk
                try:
                    frame_info_cmd = [
                        'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                        '-count_frames', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', file_path
                    ]
                    frame_result = subprocess.run(frame_info_cmd, capture_output=True, text=True, check=True)
                    frame_count = int(frame_result.stdout.strip())
                    print(f"   • {file} ({size_mb:.1f} MB, {frame_count} frames)")
                except:
                    print(f"   • {file} ({size_mb:.1f} MB)")
        
        return 0 if failed == 0 else 1
    
    else:
        # Image sequence conversion mode
        if args.files:
            # Process specific files
            print("File Converter")
            print("=" * 50)
            print(f"Processing {len(args.files)} specified files")
            print(f"Frame Rate: {args.frame_rate} fps")
            print(f"Output Directory: {args.output_dir}")
            print(f"Codec: {args.codec}")
            print(f"Preset: {args.preset}")
            print(f"CRF: {args.crf}")
            if args.start_frame is not None:
                print(f"Start Frame: {args.start_frame}")
            if args.slow_motion is not None:
                target_fps = args.frame_rate / args.slow_motion
                print(f"Slow Motion: {args.frame_rate}fps → {target_fps:.1f}fps ({args.slow_motion}x slower)")
            elif args.reverse_slow_motion is not None:
                print(f"Reverse Slow Motion: {args.reverse_slow_motion}x speed up (maintaining {args.frame_rate}fps)")
            elif args.interpolate_fps is not None:
                if args.interpolate_fps > args.frame_rate:
                    print(f"Frame Upsampling: {args.frame_rate}fps → {args.interpolate_fps}fps ({args.interpolation_mode} mode)")
                else:
                    print(f"Frame Downsampling: {args.frame_rate}fps → {args.interpolate_fps}fps (frame dropping)")
            print()
            
            if args.dry_run:
                print("Dry run mode - no files will be converted")
                print("Files that would be processed:")
                for file in args.files:
                    print(f"   • {file}")
                return 0
            
            # Process each specified file
            successful = 0
            failed = 0
            
            for file_path in args.files:
                if process_single_file(
                    file_path, args.output_dir,
                    args.frame_rate, args.codec, args.preset, args.crf, args.pixel_format, args.start_frame,
                    args.interpolate_fps, args.interpolation_mode, args.slow_motion, args.reverse_slow_motion
                ):
                    successful += 1
                else:
                    failed += 1
                print()
            
            # Summary
            print("=" * 50)
            print(f"File conversion complete!")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if successful > 0:
                print(f"\n Output files in '{args.output_dir}' directory:")
                output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.mp4')]
                for file in sorted(output_files):
                    file_path = os.path.join(args.output_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   • {file} ({size_mb:.1f} MB)")
            
            return 0 if failed == 0 else 1
        
        else:
            # Auto-detect image sequences
            sequences = find_image_sequences()
            
            if not sequences:
                print("   No valid image sequences found!")
                print("   Make sure your files follow a pattern like: name_0001.jpg, name_0002.jpg, etc.")
                print("   Or specify files directly: python convert_sequences.py file1.jpg file2.mp4")
                return 1
            
            print(f"Found {len(sequences)} image sequences:")
            for name, frames in sequences.items():
                frame_count = len(frames)
                min_frame = min(frames)
                max_frame = max(frames)
                print(f"   • {name}: {frame_count} frames ({min_frame:04d}-{max_frame:04d})")
            print()
            
            if args.dry_run:
                print("Dry run mode - no files will be converted")
                return 0
            
            # Convert each sequence
            successful = 0
            failed = 0
            
            for sequence_name, frame_numbers in sequences.items():
                if convert_sequence_to_mp4(
                    sequence_name, frame_numbers, args.output_dir,
                    args.frame_rate, args.codec, args.preset, args.crf, args.pixel_format, args.start_frame,
                    args.interpolate_fps, args.interpolation_mode, args.slow_motion, args.reverse_slow_motion
                ):
                    successful += 1
                else:
                    failed += 1
                print()
            
            # Summary
            print("=" * 50)
            print(f"Conversion complete!")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if successful > 0:
                print(f"\n Output files in '{args.output_dir}' directory:")
                output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.mp4')]
                for file in sorted(output_files):
                    file_path = os.path.join(args.output_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   • {file} ({size_mb:.1f} MB)")
            
            return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 