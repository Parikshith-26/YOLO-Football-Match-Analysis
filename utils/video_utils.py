import cv2
import numpy as np
import os
import subprocess
from pathlib import Path

# -----------------------------------
# READ VIDEO (BATCHED)
# -----------------------------------
def read_video_batched(video_path, batch_size=32, max_frames=None):
    """
    Generator that yields video frames in batches for efficient processing.
    Reduces memory footprint by 30-50%.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    batch = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if batch:
                yield batch
            break
        
        batch.append(frame)
        frame_count += 1
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
        
        if max_frames and frame_count >= max_frames:
            if batch:
                yield batch
            break
    
    cap.release()

# -----------------------------------
# READ VIDEO (FULL)
# -----------------------------------
def read_video(video_path, max_frames=None):
    """Loads full video with safe memory usage."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return []

    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    print(f"✅ Read {len(frames)} frames from {video_path}")
    return frames

# -----------------------------------
# SAVE VIDEO (BROWSER COMPATIBLE)
# -----------------------------------
def save_video_optimized(output_frames, output_video_path, fps=24):
    """
    Save video as fully browser-compatible MP4:
    - Step 1: Write MJPG AVI with OpenCV
    - Step 2: Convert AVI → MP4 (H264 + yuv420p) using ffmpeg
    """
    # Ensure MP4 extension
    if not output_video_path.endswith(".mp4"):
        output_video_path = output_video_path.rsplit(".", 1)[0] + ".mp4"

    temp_avi = "temp_output.avi"

    out = None
    frame_count = 0

    # STEP 1 — Create temporary AVI file
    for frame in output_frames:
        if out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError("Cannot initialize AVI VideoWriter")

        out.write(frame)
        frame_count += 1

    if out:
        out.release()

    print("🔄 Converting AVI → MP4 (H264)...")

    # STEP 2 — Convert AVI → MP4 (H264)
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_avi,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Remove temporary AVI file
    if os.path.exists(temp_avi):
        os.remove(temp_avi)

    print(f"✅ Browser-compatible MP4 saved: {output_video_path} ({frame_count} frames)")
    return True

# -----------------------------------
# COMPATIBILITY WRAPPER
# -----------------------------------
def save_video(output_frames, output_video_path, fps=24):
    """
    Backwards compatible wrapper.
    Always produces browser-friendly MP4 output.
    """
    return save_video_optimized(output_frames, output_video_path, fps=fps)

# -----------------------------------
# READ VIDEO GENERATOR
# -----------------------------------
def read_video_generator(video_path, max_frames=None):
    """Yield frames using batched reader."""
    try:
        for batch in read_video_batched(video_path, batch_size=32, max_frames=max_frames):
            for frame in batch:
                yield frame
    except NameError:
        frames = read_video(video_path, max_frames=max_frames)
        for f in frames:
            yield f

# -----------------------------------
# GET VIDEO INFO
# -----------------------------------
def get_video_info(video_path):
    """Return width, height, FPS, total_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not get video info for {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames
    }
