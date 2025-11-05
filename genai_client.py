import os
import json
import re
import base64
import time
from pathlib import Path
from openai import OpenAI
from PIL import Image, ImageOps
import numpy as np
from colorama import Fore, Style, init
from time import sleep
import cv2

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    print("Google GenAI imported successfully")
except ImportError as e:
    print(f"Google GenAI not available: {e}")
    GENAI_AVAILABLE = False

# === Configuration ===
GOOGLE_API_KEY = None  # Set your Google AI API Key

# Evaluation config
SAMPLE_FRAMES = 16
MODEL_NAME = "gemini-2.5-pro"
NUM_VIDEOS_PER_ITEM = 6  # Generate 6 videos per item

# Evaluation prompts
TEXT_PROMPT_1 = "Evaluate whether the following video faithfully and coherently visualizes the process described in the text."
TEXT_PROMPT_2 = """
Judge the video only based on visible evidence.
Rate each of the following five aspects on a 0–4 scale (0 = poor, 4 = excellent):

Directly formalize your output answer as a JSON format:

{
  "instruction_alignment": 0,   // How well the video follows the described structure and sequence
  "temporal_consistency": 0,    // Smoothness and continuity between frames
  "visual_stability": 0,        // Stability of appearance, motion, and viewpoint
  "content_fidelity": 0,        // Preservation of key elements without hallucination or loss
  "focus_relevance": 0          // Whether visual attention stays on the intended objects or regions
}

"""

def add_16_9_padding(image_path, color=(255, 255, 255)):
    """
    Add padding to make the image 16:9 aspect ratio

    Args:
        image_path: Image path
        color: Padding color in RGB, default white

    Returns:
        PIL.Image: Processed image object
    """
    img = Image.open(image_path)
    original_width, original_height = img.size

    target_aspect = 16 / 9  # 16:9 aspect ratio
    current_aspect = original_width / original_height

    if current_aspect > target_aspect:
        # Image too wide; add padding to top and bottom
        new_height = int(original_width / target_aspect)
        padding_height = (new_height - original_height) // 2
        padded_img = ImageOps.expand(
            img,
            border=(0, padding_height, 0, padding_height),
            fill=color
        )
    else:
        # Image too tall; add padding to left and right
        new_width = int(original_height * target_aspect)
        padding_width = (new_width - original_width) // 2
        padded_img = ImageOps.expand(
            img,
            border=(padding_width, 0, padding_width, 0),
            fill=color
        )

    return padded_img

def initialize_genai_client():
    """
    Initialize Google GenAI client
    """
    if not GENAI_AVAILABLE:
        print("Google GenAI not available")
        return None

    try:
        # Try using API Key
        if GOOGLE_API_KEY:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            print("GenAI client initialized with API key")
            return client
        else:
            return genai.Client()
        print("No valid authentication found for GenAI")
        print("Please provide GOOGLE_API_KEY or arc.json file")
        return None

    except Exception as e:
        print(f"Failed to initialize GenAI client: {e}")
        return None

def generate_video_from_image(genai_client, image_path: str, prompt: str, output_path: str, video_id: int):
    """
    Generate a video from an image using Google GenAI (adds 16:9 padding)
    """
    print(f"{Fore.CYAN}Generating video #{video_id}: {image_path} -> {output_path}{Style.RESET_ALL}")

    if not genai_client:
        print("GenAI client not available")
        return False

    # Create temp padded image path
    temp_padded_path = f"temp_padded_{video_id}_{int(time.time())}.jpg"

    try:
        # Adding 16:9 padding to image...
        print(f"📐 Adding 16:9 padding to image...")
        padded_img = add_16_9_padding(image_path, color=(255, 255, 255))  # white background

        # Save temp padded image
        padded_img.save(temp_padded_path, format='JPEG', quality=95)
        print(f"Padded image saved: {temp_padded_path}")

        # Read padded image
        with open(temp_padded_path, "rb") as f:
            image_bytes = f.read()

        # Detect image format
        mime_type = "image/jpeg"
        image = types.Image(image_bytes=image_bytes, mime_type=mime_type)
        print(f"Padded image loaded: {len(image_bytes)} bytes, {mime_type}")

        # Send generation request (removed aspect_ratio to avoid errors)
        print(f"🚀 Sending generation request #{video_id}...")
        operation = genai_client.models.generate_videos(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            image=image,
            config=types.GenerateVideosConfig(
                negative_prompt="cartoon, drawing, low quality",
            )
        )

        print(f"Request #{video_id} sent, operation ID: {operation.name}")

        # Poll until done
        print(f"⏳ Waiting for video generation #{video_id}...")
        attempt = 0
        max_attempts = 120  # 30-minute timeout

        while not operation.done and attempt < max_attempts:
            attempt += 1
            elapsed = attempt * 15
            print(f"[{attempt:3d}] Video #{video_id} generating... ({elapsed//60}m {elapsed%60}s)")
            time.sleep(15)
            operation = genai_client.operations.get(operation)

        if not operation.done:
            print(f"Video #{video_id} generation timeout")
            return False

        print(f"Video #{video_id} generation completed!")

        # Save video
        if hasattr(operation, 'response') and operation.response:
            response = operation.response

            videos = []
            if hasattr(response, 'generated_videos'):
                videos = response.generated_videos
            elif hasattr(response, 'videos'):
                videos = response.videos

            if videos:
                # Take the first video
                video = videos[0]
                video_data = None

                # Try to get video data
                if hasattr(video, 'video') and hasattr(video.video, 'video_bytes'):
                    video_data = video.video.video_bytes
                elif hasattr(video, 'video_bytes'):
                    video_data = video.video_bytes
                elif hasattr(video, 'model_dump'):
                    dump = video.model_dump()
                    if 'video' in dump and 'video_bytes' in dump['video']:
                        video_data = dump['video']['video_bytes']

                if video_data:
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(video_data)

                    size_mb = len(video_data) / (1024 * 1024)
                    print(f"Video #{video_id} saved: {output_path} ({size_mb:.1f} MB)")
                    return True
                else:
                    print(f"No video data found for video #{video_id}")
                    return False
            else:
                print(f"No videos in response for video #{video_id}")
                return False
        else:
            print(f"No response from operation for video #{video_id}")
            return False

    except Exception as e:
        print(f"Error generating video #{video_id}: {e}")
        return False

    finally:
        # Clean up temp files
        if os.path.exists(temp_padded_path):
            try:
                os.remove(temp_padded_path)
                print(f"🗑️ Cleaned up temp file: {temp_padded_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not remove temp file {temp_padded_path}: {e}")

def generate_input_images_base64(mp4_path, num_frames=16):
    """
    Extract frames from a video and convert to base64
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    results = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        results.append(buffer.tobytes())

    cap.release()
    return results

def evaluate_video(video_path: str, input_description: str, video_id: int):
    """
    Evaluate a video using VisionLabs API
    """
    print(f"{Fore.YELLOW}Evaluating video #{video_id}: {video_path}{Style.RESET_ALL}")

    try:
        eval_client = initialize_genai_client()

        # Extract video frames
        input_images = generate_input_images_base64(video_path, SAMPLE_FRAMES)

        # Build messages
        eval_prompt = f"{TEXT_PROMPT_1}\\n\\nVideo frames are provided as images.\\n\\nDescription is:\\n{input_description}\\n\\n{TEXT_PROMPT_2}"

        # Attach video frames
        input_contents = [eval_prompt]
        for frame_base64 in input_images:
            input_contents.append(types.Part.from_bytes(
                data=frame_base64,
                mime_type='image/jpeg',
            ))

        # Call evaluation API
        for attempt in range(5):
            try:
                response = eval_client.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=input_contents
                )

                result = response.text
                print(f"Evaluation #{video_id} completed")
                print(f"{Fore.YELLOW} The response from the model is: {Style.RESET_ALL}", result)
                sleep(1)
                print(result)
                return result

            except Exception as e:
                print(f"{Fore.RED}API call failed, waiting to retry (attempt {attempt+1}): {e}{Style.RESET_ALL}")
                sleep(2 ** attempt)  # exponential backoff
                continue

        print(f"{Fore.RED}All retries exhausted; call still failed{Style.RESET_ALL}")
        return f"error: Evaluation failed for video #{video_id} after retries"

    except Exception as e:
        print(f"Error evaluating video #{video_id}: {e}")
        return f"error: {str(e)}"
