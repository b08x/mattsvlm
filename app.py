#!/usr/bin/env python3
"""
Matt's VLM Pipeline

A tool for analyzing video clips or image folders using a Vision Language Model.
"""

import argparse
import sys
import os
import time
import ollama
import pathlib
import json
from datetime import datetime
from dotenv import load_dotenv
from src.video import extract_frames, get_video_metadata, load_frames_from_folder
from src.llm import process_frames
from src.utils import (
    check_dependencies, 
    check_ollama_availability, 
    validate_video_file, 
    get_config,
    extract_json_from_response,
    update_process_log
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process video clips or image folders with a Vision Language Model",
        prog="mattsvlm"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to the video file (MP4/H264) OR a folder containing images"
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        default="summarize what is happening",
        help="Text prompt for the VLM (default: 'summarize what is happening'). If using --register, this serves as additional context."
    )
    
    parser.add_argument(
        "-fps", "--frames-per-second",
        type=int,
        default=8,
        help="Frames per second to extract from video (default: 8). Ignored for image folders."
    )
    
    parser.add_argument(
        "-bs", "--batch-size",
        type=int,
        default=None,
        help="Maximum number of frames to process in each batch (default: auto-calculated)"
    )

    parser.add_argument(
        "-r", "--register",
        type=str,
        choices=['it-workflow', 'gen-ai', 'tech-support', 'educational'],
        default=None,
        help="Select a specialized register/template for analysis (e.g., 'it-workflow')."
    )
    
    parser.add_argument(
        "-o", "--output-log",
        type=str,
        default="analysis_log.json",
        help="Path to the JSON log file to append results to (default: analysis_log.json)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Explicitly load .env from the script's directory parent (project root)
    script_dir = pathlib.Path(__file__).parent.resolve()
    dotenv_path = script_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    # Start tracking overall execution time
    start_time = time.time()
    
    # Check dependencies first
    if not check_dependencies():
        print("Error: Missing required dependencies. Please install them and try again.")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get configuration
    config = get_config()
    
    # Check if Ollama is available
    if not check_ollama_availability():
        print(f"Error: Ollama is not available. Please make sure it's running and accessible at {config['ollama_host']}.")
        sys.exit(1)
    
    frames = []
    metadata = {}
    is_video = False
    
    try:
        # Determine Input Type
        if os.path.isdir(args.input_path):
            print(f"\n--- Loading images from folder: {args.input_path} ---")
            frames = load_frames_from_folder(args.input_path)
            print(f"Loaded {len(frames)} images.")
            # Mock metadata for folder mode
            metadata = {'duration': 0, 'width': 'N/A', 'height': 'N/A', 'fps': args.frames_per_second} 
        elif validate_video_file(args.input_path):
            is_video = True
            # Get video metadata
            metadata = get_video_metadata(args.input_path)
            print(f"Video: {args.input_path}")
            print(f"Duration: {metadata['duration']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}, FPS: {metadata['fps']:.2f}")
            
            if metadata['duration'] > 60:
                print(f"Error: Video duration ({metadata['duration']:.2f}s) exceeds maximum allowed (60s).")
                sys.exit(1)

            # Extract frames from video
            extraction_start = time.time()
            print(f"\n--- Extracting frames at {args.frames_per_second} fps... ---")
            frames = extract_frames(args.input_path, args.frames_per_second)
            extraction_time = time.time() - extraction_start
            print(f"Extracted {len(frames)} frames in {extraction_time:.2f} seconds.")
        else:
             print(f"Error: '{args.input_path}' is not a valid video file or directory.")
             sys.exit(1)

        print(f"Using Ollama at: {config['ollama_host']}")
        print(f"Using model: {config['ollama_model']} (128k context window)")
        
        # Process frames with VLM
        # If register is used, args.prompt acts as additional context for the template
        prompt_desc = f"Template: {args.register} + Context: {args.prompt}" if args.register else args.prompt
        print(f"\n--- Analyzing frames with prompt: '{prompt_desc}'... ---")
        
        result = process_frames(
            frames, 
            args.prompt, 
            batch_size=args.batch_size, 
            fps=args.frames_per_second,
            register=args.register
        )
        
        # Separate Content from Stats
        analysis_content = result
        performance_stats = ""
        
        if "--- Performance Statistics ---" in result:
            analysis_content, performance_stats = result.split("--- Performance Statistics ---", 1)
            analysis_content = analysis_content.strip()

        # Display Results
        print("\nAnalysis Result:")
        print("-" * 80)
        print(analysis_content)
        print("-" * 80)
        
        if performance_stats:
            print("\nPerformance Statistics:")
            print("-" * 80)
            print(f"--- Performance Statistics ---{performance_stats}")
            print("-" * 80)
        
        # Calculate total runtime
        total_runtime = time.time() - start_time
        print(f"\nTotal application runtime: {total_runtime:.2f} seconds")

        # --- Logging Logic ---
        # 1. Try to parse analysis content as JSON (if requested by prompt templates)
        parsed_result = extract_json_from_response(analysis_content)
        
        # 2. Construct log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_path": os.path.abspath(args.input_path),
            "input_type": "video" if is_video else "image_folder",
            "prompt": args.prompt,
            "register": args.register,
            "model_config": {
                "host": config['ollama_host'] if config['endpoint_type'] == 'ollama' else 'openai',
                "model": config['ollama_model'] if config['endpoint_type'] == 'ollama' else config['openai_model'],
                "temperature": config['temperature'],
                "top_p": config['top_p']
            },
            "parameters": {
                "fps": args.frames_per_second,
                "batch_size": args.batch_size,
                "frame_count": len(frames)
            },
            "runtime_seconds": round(total_runtime, 2),
            "result": parsed_result  # Can be dict (if JSON parsed) or str
        }
        
        # 3. Write to file
        update_process_log(args.output_log, log_entry)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ollama.ResponseError as e:
        print(f"Ollama Error: {e.error} (Status: {e.status_code})")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()