#!/usr/bin/env python
"""
Configuration file for the Stellar Video Editor
Simple workflow configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables from a local .env file if present
load_dotenv()

# Gemini / Google GenAI API
# Read your Google API Key from env. Do NOT hardcode secrets.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# RapidAPI ChatGPT-42 Configuration
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_URL = "https://chatgpt-42.p.rapidapi.com/gpt4"

# Shotstack API Configuration
# Get your API key from: https://shotstack.io/
SHOTSTACK_API_KEY = os.getenv("SHOTSTACK_API_KEY")
SHOTSTACK_BASE_URL = "https://api.shotstack.io/v1"

# Gradio API Endpoints for Media Analysis
GRADIO_VIDEO_API = "akhaliq/MiniCPM-V-4_5-video-chat"  # Updated MiniCPM-V-4.5 for video
GRADIO_AUDIO_API = "https://midasheng-midashenglm-7b.ms.show/"  # MidasHeng LM-7B for audio

# Application Settings
MAX_FILE_SIZE_MB = 100  # Maximum file size for upload
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

# UI Settings
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
THEME_COLOR = '#2b2b2b'

# Processing Settings
ANALYSIS_TIMEOUT = 120  # seconds
RENDER_TIMEOUT = 300    # seconds
RENDER_POLL_INTERVAL = 5  # seconds

# Video Analysis Settings
VIDEO_ANALYSIS_FPS = 3  # Frame sampling rate for video analysis
VIDEO_ANALYSIS_FORCE_PACKING = 0  # Force packing setting
VIDEO_ANALYSIS_PRIMARY_ENDPOINT = "/process_video_and_question_1"
VIDEO_ANALYSIS_FALLBACK_ENDPOINT = "/process_video_and_question"

# Logging Settings
ENABLE_VERBOSE_LOGGING = False  # Set to False to reduce API call logging
ENABLE_GRADIO_LOGGING = False   # Set to False to reduce Gradio client logging

# Asset enforcement settings
# 'remove' will drop clips that reference unknown media; 'replace' will set src to 'placeholder'
ASSET_ENFORCEMENT_MODE = os.getenv("ASSET_ENFORCEMENT_MODE", "remove")

# Prompt Enhancer settings
# Toggle to enable the PromptEnhancer stage between user input and DirectorAgent
ENABLE_PROMPT_ENHANCER = os.getenv("ENABLE_PROMPT_ENHANCER", "1") not in {"0", "false", "False"}

# Default Gemini model to use for prompt enhancement
# Use gemini-2.0-flash-exp (tested and working) or override in .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# System instruction for the enhancer agent
PROMPT_ENHANCER_INSTRUCTION = os.getenv(
    "PROMPT_ENHANCER_INSTRUCTION",
    (
        "You are a prompt Enhancer for a video editing assistant, who understands user intent, "
        "style, and how the edit should feel. Enhance the user prompt while keeping intent intact.\n"
        "- Keep output under 10 lines.\n"
        "- Use a simple paragraph style (no lists).\n"
        "- Include style, pacing, mood, rough duration/aspect if obvious, and constraints like using only provided media.\n"
    ),
)

# Loudly AI Music API
# Place LOUDLY_API_KEY in your .env. You can override the URL if needed.
LOUDLY_API_KEY = os.getenv("LOUDLY_API_KEY")
LOUDLY_API_URL = os.getenv("LOUDLY_API_URL", "https://soundtracks.loudly.com/api/ai/prompt/songs")

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"RapidAPI: {'Configured' if RAPIDAPI_KEY else 'Not configured'}")
    print(f"Shotstack: {'Configured' if SHOTSTACK_API_KEY != 'YOUR_SHOTSTACK_API_KEY' else 'Not configured (demo mode)'}")
    print(f"Gemini: {'Configured' if GOOGLE_API_KEY else 'Not configured'} | Enhancer: {'On' if ENABLE_PROMPT_ENHANCER else 'Off'}")
