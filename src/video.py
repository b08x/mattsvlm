"""
Video processing module for extracting frames from videos and loading images from folders.
"""

import cv2
import numpy as np
import base64
import os
import glob
from io import BytesIO
from PIL import Image


def extract_frames(video_path, fps=8):
    """
    Extract frames from the video at the specified fps rate.
    
    Args:
        video_path (str): Path to the video file
        fps (int, optional): Frames per second to extract. Defaults to 8.
        
    Returns:
        list: List of frames as base64-encoded strings
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    
    # Calculate frame interval for desired fps
    frame_interval = int(video_fps / fps)
    
    # Check if the video is too long (more than 60 seconds)
    if duration > 60:
        raise ValueError(f"Video duration ({duration:.2f}s) exceeds maximum allowed (60s)")
    
    frames = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only process frames at the desired interval
        if frame_number % frame_interval == 0:
            # Convert from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to base64 for the API
            pil_img = Image.fromarray(rgb_frame)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            frames.append(img_str)
        
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    
    if not frames:
        raise ValueError("No frames were extracted from the video")
    
    return frames


def load_frames_from_folder(folder_path, max_frames=60):
    """
    Load images from a folder to simulate a video sequence or screenshot analysis.
    
    Args:
        folder_path (str): Path to the folder containing images
        max_frames (int): Maximum number of images to load (to prevent overloading)
        
    Returns:
        list: List of frames as base64-encoded strings
    """
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        # Also check uppercase
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    # Sort files to ensure order (important for sequence analysis)
    image_files.sort()
    
    if not image_files:
        raise ValueError(f"No valid image files found in {folder_path}")
    
    if len(image_files) > max_frames:
        print(f"Warning: Found {len(image_files)} images, limiting to first {max_frames}.")
        image_files = image_files[:max_frames]
        
    frames = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary (e.g. for PNGs with alpha channel)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                frames.append(img_str)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            
    if not frames:
        raise ValueError("Failed to load any valid frames from folder")
        
    return frames


def get_video_metadata(video_path):
    """
    Get metadata for the video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Release the video capture object
    cap.release()
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration
    }