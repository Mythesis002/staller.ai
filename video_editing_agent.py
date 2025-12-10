#!/usr/bin/env python
"""
Modular Video Editing Components - Clean Architecture
Following the workflow diagram: INPUT → MEDIA_ANALYSER → LLM → EXTRACTOR → SHOTSTACK → GUI
"""

import os
import time
import datetime
import requests
from requests.adapters import HTTPAdapter
try:
    # Prefer urllib3 from requests vendored location if available
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from gradio_client import Client, handle_file

from config import (
    SHOTSTACK_API_KEY, SHOTSTACK_BASE_URL,
    GRADIO_VIDEO_API, GRADIO_AUDIO_API, ANALYSIS_TIMEOUT, RENDER_TIMEOUT,
    VIDEO_ANALYSIS_FPS, VIDEO_ANALYSIS_FORCE_PACKING, 
    VIDEO_ANALYSIS_PRIMARY_ENDPOINT, VIDEO_ANALYSIS_FALLBACK_ENDPOINT,
    GOOGLE_API_KEY, GEMINI_MODEL, PROMPT_ENHANCER_INSTRUCTION, ENABLE_PROMPT_ENHANCER,
    LOUDLY_API_KEY, LOUDLY_API_URL,
)
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


@dataclass
class MediaAnalysis:
    """Structure for media analysis results"""
    file_path: str
    file_type: str
    filename: str
    analysis: str
    metadata: Dict[str, Any]
    timestamp: str
    status: str = "success"
    cloud_url: str = ""  # Cloudinary or external URL for this media file


@dataclass
class EditingPlan:
    """Structure for editing plan with JSON timeline"""
    content: str  # Natural language explanation
    json_plan: Dict[str, Any]  # Shotstack-compatible JSON
    media_files: List[str]
    style: str
    duration: float
    timestamp: str
    plan_id: str


class PromptEnhancer:
    """Enhances the user prompt using Google's Gemini API.
    Falls back to returning the original prompt if the API/key is unavailable or any error occurs.
    """

    def __init__(self):
        self.available = False
        self._client = None
        # Lazy import and client init to avoid hard dependency
        try:
            if GOOGLE_API_KEY:
                from google import genai  # type: ignore
                import os as _os
                if not _os.environ.get("GOOGLE_API_KEY"):
                    _os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                self._client = genai.Client(api_key=GOOGLE_API_KEY)
                self.available = True
                print("🧠 PromptEnhancer: Gemini client initialized")
            else:
                print("⚠️ PromptEnhancer: GOOGLE_API_KEY not set; enhancer will pass-through")
        except Exception as e:
            print(f"⚠️ PromptEnhancer init failed: {e}")
            self.available = False

    def _summarize_media_context(self, analyses: Optional[Dict[str, Any]]) -> str:
        try:
            if not analyses or not isinstance(analyses, dict):
                return ""
            parts = []
            v = a = i = 0
            for k, vdict in analyses.items():
                if not isinstance(vdict, dict):
                    continue
                ftype = vdict.get("file_type") or vdict.get("type")
                if ftype == "video": v += 1
                elif ftype == "audio": a += 1
                elif ftype == "image": i += 1
            if v: parts.append(f"{v} video{'s' if v>1 else ''}")
            if a: parts.append(f"{a} audio file{'s' if a>1 else ''}")
            if i: parts.append(f"{i} image{'s' if i>1 else ''}")
            return f"Media context: {', '.join(parts)}." if parts else ""
        except Exception:
            return ""

    def enhance(self, user_prompt: str, analyses: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "model": GEMINI_MODEL,
            "enabled": bool(ENABLE_PROMPT_ENHANCER),
            "used": False,
        }
        if not ENABLE_PROMPT_ENHANCER:
            return user_prompt, meta
        if not self.available or not self._client:
            return user_prompt, meta

        try:
            media_ctx = self._summarize_media_context(analyses)
            contents = f"{PROMPT_ENHANCER_INSTRUCTION}\n\n{media_ctx}\n\nUser prompt: {user_prompt}"
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config={
                    "temperature": 0.9,
                    "top_p": 0.95,
                },
            )
            enhanced = (getattr(response, "text", None) or "").strip()
            if not enhanced:
                return user_prompt, meta
            meta["used"] = True
            return enhanced, meta
        except Exception as e:
            meta["error"] = str(e)
            return user_prompt, meta

class GeminiImageAnalyzer:
    """Analyzes images using Google's Gemini vision model.
    Falls back to basic metadata analysis if the API/key is unavailable or any error occurs.
    """

    def __init__(self):
        self.available = False
        self._client = None
        # Lazy import and client init to avoid hard dependency
        try:
            if GOOGLE_API_KEY:
                from google import genai  # type: ignore
                import os as _os
                if not _os.environ.get("GOOGLE_API_KEY"):
                    _os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                self._client = genai.Client(api_key=GOOGLE_API_KEY)
                self.available = True
                print("🖼️ GeminiImageAnalyzer: Gemini vision client initialized")
            else:
                print("⚠️ GeminiImageAnalyzer: GOOGLE_API_KEY not set; will use basic analysis")
        except Exception as e:
            print(f"⚠️ GeminiImageAnalyzer init failed: {e}")
            self.available = False

    def analyze(self, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze image content using Gemini vision model.
        Returns (analysis_text, metadata_dict)
        """
        metadata: Dict[str, Any] = {
            "analysis_method": "gemini-vision" if self.available else "basic_metadata",
            "model": GEMINI_MODEL if self.available else None,
            "used": False,
        }

        if not self.available or not self._client:
            return self._get_basic_analysis(file_path, filename), metadata

        try:
            print(f"🖼️ Analyzing image with Gemini: {filename}")
            
            # Structured analysis prompt for professional video editing context
            analysis_prompt = """Analyze this image in detail for professional video editing. Provide a structured breakdown:

1. Visual Content - Describe the main subjects, objects, people, composition, and framing.
2. Colors & Lighting - Analyze color palette, dominant colors, lighting quality, and mood.
3. Style & Aesthetic - Identify artistic style, genre, and visual treatment.
4. Technical Quality - Assess resolution, sharpness, and production quality.
5. Emotional Tone - Explain the mood, atmosphere, and emotional impact.
6. Editing Opportunities - Suggest how to use this in video editing (backgrounds, overlays, transitions, title cards, color grading reference, etc.).
7. Visual Storytelling - What story or message does this image convey?

Output in a clear, labeled, structured format suitable for professional post-production."""

            # Upload the image file
            from google.genai import types  # type: ignore
            
            # Read the image file
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Create file part for the API
            file_part = types.Part.from_bytes(
                data=image_data,
                mime_type=self._get_mime_type(file_path)
            )
            
            # Make API call with vision model
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[analysis_prompt, file_part],
                config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                },
            )
            
            analysis_text = (getattr(response, "text", None) or "").strip()
            
            if not analysis_text:
                print("⚠️ Gemini returned empty response, using basic analysis")
                return self._get_basic_analysis(file_path, filename), metadata
            
            metadata["used"] = True
            print("✅ Image analysis completed with Gemini")
            
            # Format the analysis with header
            formatted_analysis = f"""📸 **AI Image Analysis for {filename}**

{analysis_text}

**Analysis Method:** Gemini Vision AI ({GEMINI_MODEL})"""
            
            return formatted_analysis, metadata
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini image analysis failed: {error_msg}")
            metadata["error"] = error_msg
            return self._get_basic_analysis(file_path, filename), metadata

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for image file"""
        ext = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
        }
        return mime_types.get(ext, 'image/jpeg')

    def _get_basic_analysis(self, file_path: str, filename: str) -> str:
        """Fallback to basic metadata analysis"""
        try:
            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            analysis = f"""📸 **Image Analysis for {filename}**

**File Information:**
• Format: {file_ext.upper()} image file
• Size: {file_size / (1024 * 1024):.1f} MB

**Usage Recommendations:**
• Can be used as background, overlay, or thumbnail
• Suitable for title cards, transitions, or visual elements
• Good for creating slideshow-style content
• Can serve as reference material for color grading

**Technical Assessment:**
• Standard image format compatible with video editing
• {'High resolution' if file_size > 1024*1024 else 'Standard resolution'} suitable for video use
• Ready for timeline integration

**Editing Suggestions:**
• Use as static background with text overlays
• Apply ken burns effect for dynamic movement
• Incorporate into montage sequences
• Use for thumbnail generation

**Note:** AI vision analysis unavailable. Using basic metadata."""
            return analysis
        except Exception as e:
            return f"Image file detected: {filename}. Error: {str(e)}"


class MusicGenerator:
    """Interface for AI music generation. Replace generate() with actual implementation."""

    def __init__(self):
        self.api_key = LOUDLY_API_KEY
        self.url = LOUDLY_API_URL
        self.available = bool(self.api_key and self.url)

    def generate(self, music_script: str) -> Optional[str]:
        try:
            if not music_script or not music_script.strip():
                print("⚠️ MusicGenerator: Empty music_script provided")
                return None
            if not self.available:
                print("⚠️ MusicGenerator: Not available (check LOUDLY_API_KEY in .env)")
                return None
            
            print(f"🎵 Calling Loudly API at {self.url}...")
            headers = {
                "Accept": "application/json",
                "API-KEY": self.api_key,
            }
            # Use multipart form to match provider expectations
            files = {
                "prompt": (None, music_script.strip()),
                "duration": (None, "20"),
                # Optional fields accepted by API; send empty if not specified
                "test": (None, ""),
                "structure_id": (None, ""),
            }

            # Configure session with retries/backoff for transient issues
            session = requests.Session()
            if Retry is not None:
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"],
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("https://", adapter)
                session.mount("http://", adapter)

            resp = session.post(self.url, headers=headers, files=files, timeout=(10, 60))
            print(f"🎵 Loudly API response: HTTP {resp.status_code}")
            
            if resp.status_code >= 400:
                try:
                    print(f"⚠️ Loudly API HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception:
                    pass
                return None
            
            j = resp.json() if hasattr(resp, "json") else {}
            print(f"🎵 Response JSON keys: {list(j.keys())}")
            url = j.get("music_file_path") or j.get("url") or j.get("music_url")
            
            if isinstance(url, str) and url.startswith("http"):
                print(f"✅ Music URL extracted: {url}")
                return url
            else:
                print(f"⚠️ No valid music URL in response. Got: {url}")
                return None
        except requests.exceptions.RequestException as re:
            print(f"❌ Loudly API request error: {re}")
            return None
        except Exception as e:
            print(f"❌ Loudly API unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None

class MediaAnalyzer:
    """Handles analysis of video, audio, and image files"""
    
    def __init__(self):
        self.video_client = None
        self.audio_client = None
        self.image_analyzer = GeminiImageAnalyzer()  # Initialize Gemini image analyzer
        self.client_cache = {}  # Cache clients to avoid repeated initialization
        self.last_analysis_time = {}  # Track last analysis time to avoid rapid calls
        
    def analyze_video(self, file_path: str) -> MediaAnalysis:
        """Analyze video content using MiniCPM-V-4.5 via akhaliq/MiniCPM-V-4_5-video-chat"""
        filename = os.path.basename(file_path)
        
        try:
            # Initialize video client if needed
            if not self.video_client:
                print(f"🔗 Connecting to video analysis API: {GRADIO_VIDEO_API}")
                # Increase timeout for large video uploads
                import httpx
                self.video_client = Client(
                    GRADIO_VIDEO_API,
                    httpx_kwargs={"timeout": httpx.Timeout(300.0, connect=60.0)}  # 5 min total, 1 min connect
                )
                print(f"✅ Video client initialized successfully")
            
            print(f"🎥 Analyzing video: {filename}")
            
            # Structured analysis question for the new API
            analysis_question = """Analyze this video and return a structured, professional breakdown for an expert After Effects editor who won’t watch the video. Include:

1. Visual Content – Describe scenes, objects, people, composition, and lighting.
2. Actions & Movements – List key movements, gestures, and camera actions.
3. Audio – Summarize dialogue, music (genre, mood, tempo), and sound effects.
4. Emotional Tone – Explain mood, energy, and emotional atmosphere.
5. Technical Quality – Evaluate resolution, lighting, color, and stability.
6. Key Moments – Identify major scenes, transitions, and highlight points.
7. Story Flow – Summarize narrative, pacing, and structure.
8. Editing Opportunities – Suggest pro-level cuts, transitions, effects, VFX, and color grading ideas.

Output in a clear, labeled, structured format suitable for professional post-production.
"""

            # Prepare video file
            video_file = handle_file(file_path)
            
            # Make API call with new format
            print("📡 Sending video to AI analysis service...")
            result = self.video_client.predict(
                video={"video": video_file},
                question=analysis_question,
                fps=VIDEO_ANALYSIS_FPS,
                force_packing=VIDEO_ANALYSIS_FORCE_PACKING,
                history=[],
                api_name=VIDEO_ANALYSIS_PRIMARY_ENDPOINT
            )
            
            print("✅ Video analysis completed successfully")
            
            # Extract metadata
            file_size = os.path.getsize(file_path)
            metadata = {
                "file_size_mb": file_size / (1024 * 1024),
                "file_extension": Path(file_path).suffix.lower(),
                "analysis_method": "MiniCPM-V-4.5-video-chat",
                "api_endpoint": GRADIO_VIDEO_API,
                "fps_used": 3
            }
            
            return MediaAnalysis(
                file_path=file_path,
                file_type="video",
                filename=filename,
                analysis=str(result),
                metadata=metadata,
                timestamp=datetime.datetime.now().isoformat(),
                status="success",
                cloud_url=""  # Will be injected later if url_mappings provided
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Video analysis failed: {error_msg}")
            
            # Check for specific API errors and provide helpful messages
            if "Could not fetch config" in error_msg:
                print("🔧 API configuration error - the service endpoint may have changed")
                print("💡 Falling back to basic analysis mode")
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print("⏰ API timeout - the video file might be too large or the service is busy")
                print("💡 Using fallback analysis with basic video metadata")
            elif "parameter" in error_msg.lower() or "api_name" in error_msg.lower():
                print("🔧 API parameter error - trying alternative endpoint...")
                # Try alternative API endpoint
                try:
                    result = self.video_client.predict(
                        video={"video": handle_file(file_path)},
                        question=analysis_question,
                        fps=VIDEO_ANALYSIS_FPS,
                        force_packing=VIDEO_ANALYSIS_FORCE_PACKING,
                        history=[],
                        api_name=VIDEO_ANALYSIS_FALLBACK_ENDPOINT
                    )
                    print("✅ Video analysis completed with alternative endpoint")
                    
                    file_size = os.path.getsize(file_path)
                    metadata = {
                        "file_size_mb": file_size / (1024 * 1024),
                        "file_extension": Path(file_path).suffix.lower(),
                        "analysis_method": "MiniCPM-V-4.5-video-chat-alt"
                    }
                    
                    return MediaAnalysis(
                        file_path=file_path,
                        file_type="video",
                        filename=filename,
                        analysis=str(result),
                        metadata=metadata,
                        timestamp=datetime.datetime.now().isoformat(),
                        status="success",
                        cloud_url=""  # Will be injected later if url_mappings provided
                    )
                except Exception as alt_e:
                    print(f"❌ Alternative endpoint also failed: {str(alt_e)}")
            elif "connection" in error_msg.lower():
                print("🌐 Connection error - check internet connectivity")
            elif "gradio_client" in error_msg.lower():
                print("🔧 Gradio client error - API service may be unavailable")
            
            # Reset client to try fresh connection next time
            self.video_client = None
            
            return self._get_fallback_video_analysis(file_path, filename, error_msg)
    
    def analyze_audio(self, file_path: str) -> MediaAnalysis:
        """Analyze audio content using MidasHeng LM-7B with optimized connection handling"""
        filename = os.path.basename(file_path)
        
        # Check if we've analyzed this file recently (within 5 minutes)
        current_time = datetime.datetime.now()
        file_key = f"{filename}_{os.path.getmtime(file_path)}"
        
        if file_key in self.last_analysis_time:
            time_diff = (current_time - self.last_analysis_time[file_key]).total_seconds()
            if time_diff < 300:  # 5 minutes
                print(f"⚡ Using recent analysis for {filename} (analyzed {time_diff:.0f}s ago)")
                return self._get_cached_analysis(file_path, filename)
        
        try:
            # Use optimized client initialization
            if not self.audio_client:
                print(f"🔗 Connecting to audio analysis API: {GRADIO_AUDIO_API}")
                
                # Reduce Gradio client logging to minimize console output
                import logging
                gradio_logger = logging.getLogger("gradio_client")
                httpx_logger = logging.getLogger("httpx")
                original_gradio_level = gradio_logger.level
                original_httpx_level = httpx_logger.level
                
                gradio_logger.setLevel(logging.ERROR)
                httpx_logger.setLevel(logging.ERROR)
                
                self.audio_client = Client(GRADIO_AUDIO_API)
                
                # Restore original logging levels
                gradio_logger.setLevel(original_gradio_level)
                httpx_logger.setLevel(original_httpx_level)
                
                print(f"✅ Audio client initialized successfully")
            
            print(f"🎵 Analyzing audio: {filename}")
            
            # Optimized audio analysis prompt (shorter to reduce processing time)
            prompt = """Analyze this audio for video editing:

**Genre & Style**: Music type, cultural influences
**Tempo & Energy**: BPM, energy level, mood
**Structure**: Key sections, transitions, beat drops
**Quality**: Audio clarity and production quality
**Editing Tips**: Best cut points, sync opportunities
**Visual Match**: How visuals should complement the audio

Keep analysis concise but detailed for professional video editing."""

            audio_file = handle_file(file_path)
            
            # Make single API call (remove retry mechanism to reduce calls)
            print("📡 Sending audio to AI analysis service...")
            
            result = self.audio_client.predict(
                prompt,
                audio_file,
                api_name="/infer"
            )
            
            print("✅ Audio analysis completed successfully")
            
            # Update last analysis time
            self.last_analysis_time[file_key] = current_time
            
            # Extract metadata
            file_size = os.path.getsize(file_path)
            metadata = {
                "file_size_mb": file_size / (1024 * 1024),
                "file_extension": Path(file_path).suffix.lower(),
                "analysis_method": "MidasHeng-LM-7B-Optimized",
                "cached": False
            }
            
            analysis = MediaAnalysis(
                file_path=file_path,
                file_type="audio",
                filename=filename,
                analysis=str(result),
                metadata=metadata,
                timestamp=datetime.datetime.now().isoformat(),
                status="success"
            )
            
            # Cache the analysis
            self.client_cache[file_key] = analysis
            
            return analysis
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Audio analysis failed: {error_msg}")
            
            # Check for specific API errors and provide helpful messages
            if "Could not fetch config" in error_msg:
                print("🔧 API configuration error - the audio service endpoint may have changed")
                print("💡 Falling back to basic analysis mode")
            elif "timeout" in error_msg.lower():
                print("⏰ API timeout - the audio file might be too large or the service is busy")
            elif "parameter" in error_msg.lower():
                print("🔧 API parameter error - the service API has changed")
            elif "connection" in error_msg.lower():
                print("🌐 Connection error - check internet connectivity")
            elif "key-word argument" in error_msg.lower():
                print("🔧 API signature changed - removing unsupported parameters")
            elif "gradio_client" in error_msg.lower():
                print("🔧 Gradio client error - API service may be unavailable")
            
            # Don't reset client immediately - it might recover
            # Only reset if it's a connection error
            if "connection" in error_msg.lower() or "config" in error_msg.lower():
                self.audio_client = None
            
            return self._get_fallback_audio_analysis(file_path, filename, error_msg)
    
    def analyze_image(self, file_path: str) -> MediaAnalysis:
        """Analyze image content using Gemini Vision AI"""
        filename = os.path.basename(file_path)
        
        try:
            print(f"🖼️ Analyzing image: {filename}")
            
            # Use Gemini image analyzer (with automatic fallback to basic analysis)
            analysis_text, analysis_metadata = self.image_analyzer.analyze(file_path, filename)
            
            # Get file metadata
            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            # Merge metadata
            metadata = {
                "file_size_mb": file_size / (1024 * 1024),
                "file_extension": file_ext,
                **analysis_metadata  # Include Gemini metadata (analysis_method, model, used, etc.)
            }
            
            status = "success" if analysis_metadata.get("used") else "fallback"
            
            return MediaAnalysis(
                file_path=file_path,
                file_type="image",
                filename=filename,
                analysis=analysis_text,
                metadata=metadata,
                timestamp=datetime.datetime.now().isoformat(),
                status=status,
                cloud_url=""  # Will be injected later if url_mappings provided
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Image analysis failed: {error_msg}")
            
            # Fallback to basic analysis
            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            fallback_analysis = f"""📸 **Image Analysis for {filename}** (Error Recovery)

**File Information:**
• Format: {file_ext.upper()} image file
• Size: {file_size / (1024 * 1024):.1f} MB

**Status:** Image file detected and ready for use in video editing.

**Error:** {error_msg}"""
            
            return MediaAnalysis(
                file_path=file_path,
                file_type="image",
                filename=filename,
                analysis=fallback_analysis,
                metadata={
                    "file_size_mb": file_size / (1024 * 1024),
                    "file_extension": file_ext,
                    "analysis_method": "error_fallback",
                    "error": error_msg
                },
                timestamp=datetime.datetime.now().isoformat(),
                status="error",
                cloud_url=""  # Will be injected later if url_mappings provided
            )
    
    def _get_fallback_video_analysis(self, file_path: str, filename: str, error: str) -> MediaAnalysis:
        """Provide fallback video analysis when API fails - with enhanced metadata extraction"""
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        file_ext = Path(file_path).suffix.lower()
        
        # Try to extract basic video info using OpenCV or PIL
        duration_str = "Unknown"
        resolution_str = "Unknown"
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    duration = frame_count / fps
                    duration_str = f"{duration:.1f}s"
                resolution_str = f"{width}x{height}"
                cap.release()
        except Exception:
            pass  # OpenCV not available or failed
        
        analysis = f"""📹 **Basic Video Analysis for {filename}** (Fallback Mode)

**File Information:**
• Format: {file_ext.upper()} video file
• Size: {file_size_mb:.1f} MB
• Duration: {duration_str}
• Resolution: {resolution_str}
• Status: Ready for editing (detailed AI analysis unavailable)

**Technical Assessment:**
• Standard video format compatible with editing
• Size suggests {'high quality' if file_size_mb > 50 else 'standard quality'} content
• Suitable for professional editing workflows
• File is valid and can be used in timeline

**Editing Recommendations:**
• Import as primary video track
• Apply standard color correction and grading
• Use for {'cinematic' if file_size_mb > 100 else 'social media'} style edits
• Add transitions and effects as needed
• Sync with audio if separate audio files present

**Note:** AI content analysis unavailable due to: {error}
Using basic file metadata. The video can still be edited successfully."""

        metadata = {
            "file_size_mb": file_size_mb,
            "file_extension": file_ext,
            "analysis_method": "fallback",
            "error": error
        }
        
        return MediaAnalysis(
            file_path=file_path,
            file_type="video",
            filename=filename,
            analysis=analysis,
            metadata=metadata,
            timestamp=datetime.datetime.now().isoformat(),
            status="fallback",
            cloud_url=""  # Will be injected later if url_mappings provided
        )
    
    def _get_cached_analysis(self, file_path: str, filename: str) -> MediaAnalysis:
        """Get cached analysis for a file"""
        file_key = f"{filename}_{os.path.getmtime(file_path)}"
        
        if file_key in self.client_cache:
            cached_analysis = self.client_cache[file_key]
            # Update metadata to indicate it's cached
            cached_analysis.metadata["cached"] = True
            return cached_analysis
        
        # If not in cache, return a basic cached analysis
        return MediaAnalysis(
            file_path=file_path,
            file_type="audio",
            filename=filename,
            analysis="Cached analysis not available. Please re-analyze the file.",
            metadata={"cached": True, "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)},
            timestamp=datetime.datetime.now().isoformat(),
            status="cached",
            cloud_url=""  # Will be injected later if url_mappings provided
        )

    def _get_fallback_audio_analysis(self, file_path: str, filename: str, error: str) -> MediaAnalysis:
        """Provide fallback audio analysis when API fails"""
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        file_ext = Path(file_path).suffix.lower()
        
        analysis = f"""🎵 **Basic Audio Analysis for {filename}** (Fallback Mode)

**File Information:**
• Format: {file_ext.upper()} audio file
• Size: {file_size_mb:.1f} MB
• Status: Ready for editing (detailed AI analysis unavailable)

**Technical Assessment:**
• Standard audio format compatible with editing
• {'Likely high quality' if file_size_mb > 10 else 'Standard quality'} audio
• Suitable for background music or dialogue
• File is valid and can be used in timeline

**Editing Recommendations:**
• Import as audio track
• Apply volume normalization
• Use for background music or voiceover
• Adjust levels to match video content

**Note:** AI content analysis unavailable due to: {error}
Using basic file metadata. The audio can still be edited successfully."""

        metadata = {
            "file_size_mb": file_size_mb,
            "file_extension": file_ext,
            "analysis_method": "fallback",
            "error": error
        }
        
        return MediaAnalysis(
            file_path=file_path,
            file_type="audio",
            filename=filename,
            analysis=analysis,
            metadata=metadata,
            timestamp=datetime.datetime.now().isoformat(),
            status="fallback",
            cloud_url=""  # Will be injected later if url_mappings provided
        )


class ShotstackRenderer:
    """Handles video rendering using Shotstack API"""

    
    def __init__(self):
        self.api_key = SHOTSTACK_API_KEY
        self.base_url = SHOTSTACK_BASE_URL
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def render_video(self, json_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Submit video for rendering and return render info"""
        if self.api_key == "YOUR_SHOTSTACK_API_KEY":
            return {
                "status": "demo_mode",
                "message": "Demo mode - Shotstack API key not configured",
                "render_id": f"demo_{int(time.time())}",
                "video_url": None
            }
        
        try:
            response = requests.post(
                f"{self.base_url}/render",
                headers=self.headers,
                json=json_plan,
                timeout=60
            )
            
            if response.status_code == 201:
                result = response.json()
                return {
                    "status": "submitted",
                    "render_id": result.get("response", {}).get("id"),
                    "message": "Video submitted for rendering",
                    "video_url": None
                }
            else:
                return {
                    "status": "error",
                    "message": f"Render submission failed: {response.text}",
                    "render_id": None,
                    "video_url": None
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Render error: {str(e)}",
                "render_id": None,
                "video_url": None
            }
    
    def check_render_status(self, render_id: str) -> Dict[str, Any]:
        """Check the status of a render job"""
        if self.api_key == "YOUR_SHOTSTACK_API_KEY":
            return {
                "status": "demo_mode",
                "progress": 100,
                "video_url": "https://example.com/demo_video.mp4",
                "message": "Demo mode - no actual rendering"
            }
        
        try:
            response = requests.get(
                f"{self.base_url}/render/{render_id}",
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                render_data = result.get("response", {})
                
                return {
                    "status": render_data.get("status", "unknown"),
                    "progress": render_data.get("progress", 0),
                    "video_url": render_data.get("url"),
                    "message": f"Render {render_data.get('status', 'unknown')}"
                }
            else:
                return {
                    "status": "error",
                    "progress": 0,
                    "video_url": None,
                    "message": f"Status check failed: {response.text}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "progress": 0,
                "video_url": None,
                "message": f"Status check error: {str(e)}"
            }


class MemoryManager:
    """Manages persistent storage of analyses and editing plans with context awareness"""
    
    def __init__(self, memory_dir: str = "agent_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Storage files
        self.analyses_file = self.memory_dir / "media_analyses.json"
        self.plans_file = self.memory_dir / "editing_plans.json"
        self.preferences_file = self.memory_dir / "user_preferences.json"
        self.sessions_file = self.memory_dir / "editing_sessions.json"
        self.refinements_file = self.memory_dir / "refinement_history.json"
        
        # Load existing data
        self.analyses = self._load_json(self.analyses_file, {})
        self.plans = self._load_json(self.plans_file, {})
        self.preferences = self._load_json(self.preferences_file, {})
        self.sessions = self._load_json(self.sessions_file, {})
        self.refinements = self._load_json(self.refinements_file, {})
        
        # Current session context
        self.current_session_id = None
        self.current_media_context = {}
        self.current_plan_context = {}
    
    def _normalize_path(self, p: str) -> str:
        """Normalize file path to a consistent absolute POSIX style for deduping."""
        try:
            return str(Path(p).resolve().as_posix())
        except Exception:
            return os.path.abspath(p).replace('\\', '/')
    
    def _load_json(self, file_path: Path, default: Any) -> Any:
        """Load JSON file with fallback to default"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
        return default
    
    def _save_json(self, file_path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving {file_path}: {e}")
    
    def store_analysis(self, analysis: MediaAnalysis):
        """Store media analysis in memory"""
        key = f"{analysis.filename}_{analysis.timestamp}"
        self.analyses[key] = asdict(analysis)
        self._save_json(self.analyses_file, self.analyses)
        print(f"💾 Stored analysis for {analysis.filename}")
    
    def store_plan(self, plan: EditingPlan):
        """Store editing plan in memory"""
        self.plans[plan.plan_id] = asdict(plan)
        self._save_json(self.plans_file, self.plans)
        print(f"💾 Stored editing plan {plan.plan_id}")
    
    def get_recent_analyses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent media analyses"""
        sorted_analyses = sorted(
            self.analyses.values(),
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        return sorted_analyses[:limit]
    
    def get_analysis_by_url(self, url: str) -> Optional[MediaAnalysis]:
        """
        Get analysis by cloud URL (most reliable lookup method).
        Returns MediaAnalysis object or None if not found.
        """
        matching = [
            (key, analysis) for key, analysis in self.analyses.items()
            if analysis.get('cloud_url') == url or analysis.get('file_path') == url
        ]
        
        if matching:
            latest_key, latest_analysis = max(matching, key=lambda x: x[1].get('timestamp', ''))
            return MediaAnalysis(
                file_path=latest_analysis.get('file_path', ''),
                file_type=latest_analysis.get('file_type', 'unknown'),
                filename=latest_analysis.get('filename', ''),
                analysis=latest_analysis.get('analysis', ''),
                metadata=latest_analysis.get('metadata', {}),
                timestamp=latest_analysis.get('timestamp', ''),
                status=latest_analysis.get('status', 'success'),
                cloud_url=latest_analysis.get('cloud_url', '')
            )
        return None
    
    def get_latest_analysis(self, filename: str) -> Optional[MediaAnalysis]:
        """
        Get the most recent analysis for a given filename.
        Returns MediaAnalysis object or None if not found.
        """
        matching = [
            (key, analysis) for key, analysis in self.analyses.items()
            if analysis.get('filename') == filename
        ]
        
        if not matching:
            return None
        
        # Get the most recent one
        latest_key, latest_analysis = max(matching, key=lambda x: x[1].get('timestamp', ''))
        
        # Convert dict back to MediaAnalysis object
        return MediaAnalysis(
            file_path=latest_analysis.get('file_path', ''),
            file_type=latest_analysis.get('file_type', 'unknown'),
            filename=latest_analysis.get('filename', filename),
            analysis=latest_analysis.get('analysis', ''),
            metadata=latest_analysis.get('metadata', {}),
            timestamp=latest_analysis.get('timestamp', ''),
            status=latest_analysis.get('status', 'success'),
            cloud_url=latest_analysis.get('cloud_url', '')
        )
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get specific editing plan"""
        return self.plans.get(plan_id)
    
    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        self.preferences.update(preferences)
        self._save_json(self.preferences_file, self.preferences)
        print(f"💾 Updated user preferences")
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        return self.preferences.copy()
    
    def start_editing_session(self, media_files: List[str], initial_prompt: str) -> str:
        """Start a new editing session with context tracking"""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Create session context
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "media_files": [self._normalize_path(f) for f in media_files],
            "initial_prompt": initial_prompt,
            "refinements": [],
            "plans_generated": [],
            "current_plan": None
        }
        
        self.sessions[session_id] = session_data
        self.current_session_id = session_id
        self._save_json(self.sessions_file, self.sessions)
        
        print(f"🎬 Started editing session: {session_id}")
        return session_id
    
    def add_refinement_to_session(self, session_id: str, refinement_prompt: str, 
                                  original_plan: Dict[str, Any], refined_plan: Dict[str, Any]):
        """Add a refinement to the current session"""
        if session_id not in self.sessions:
            print(f"⚠️ Session {session_id} not found")
            return
        
        refinement_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "refinement_prompt": refinement_prompt,
            "original_plan_id": original_plan.get("plan_id"),
            "refined_plan_id": refined_plan.get("plan_id"),
            "changes_made": self._analyze_plan_differences(original_plan, refined_plan)
        }
        
        self.sessions[session_id]["refinements"].append(refinement_data)
        self.sessions[session_id]["current_plan"] = refined_plan.get("plan_id")
        
        # Also store in refinements history
        refinement_id = f"{session_id}_{len(self.sessions[session_id]['refinements'])}"
        self.refinements[refinement_id] = refinement_data
        
        self._save_json(self.sessions_file, self.sessions)
        self._save_json(self.refinements_file, self.refinements)
        
        print(f"📝 Added refinement to session {session_id}")
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get complete context for a session including media analysis and plans"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        # Get media analyses for this session
        media_analyses = {}
        for file_path in session.get("media_files", []):
            filename = os.path.basename(file_path)
            for key, analysis in self.analyses.items():
                if analysis.get("filename") == filename:
                    media_analyses[filename] = analysis
                    break
        
        # Get all plans for this session
        session_plans = []
        for plan_id in session.get("plans_generated", []):
            if plan_id in self.plans:
                session_plans.append(self.plans[plan_id])
        
        return {
            "session_info": session,
            "media_analyses": media_analyses,
            "plans": session_plans,
            "refinement_history": session.get("refinements", []),
            "current_plan_id": session.get("current_plan")
        }
    
    def build_refinement_context(self, session_id: str, new_prompt: str) -> str:
        """Build comprehensive context for LLM refinement requests"""
        context = self.get_session_context(session_id)
        
        if not context:
            return new_prompt
        
        # Build context string
        context_str = f"""
EDITING SESSION CONTEXT:
========================

ORIGINAL REQUEST: {context['session_info'].get('initial_prompt', 'N/A')}

MEDIA ANALYSIS:
{self._format_media_analysis_for_context(context['media_analyses'])}

PREVIOUS EDITING DECISIONS:
{self._format_plans_for_context(context['plans'])}

REFINEMENT HISTORY:
{self._format_refinements_for_context(context['refinement_history'])}

NEW REFINEMENT REQUEST: {new_prompt}

Please create a refined editing plan that:
1. Maintains the core media analysis insights
2. Builds upon previous editing decisions where appropriate
3. Incorporates the new refinement requirements
4. Ensures consistency with the established style and mood
"""
        
        return context_str
    
    def _analyze_plan_differences(self, original: Dict[str, Any], refined: Dict[str, Any]) -> List[str]:
        """Analyze differences between two plans"""
        changes = []
        
        # Compare key aspects
        if original.get("style") != refined.get("style"):
            changes.append(f"Style changed: {original.get('style')} → {refined.get('style')}")
        
        if original.get("pacing") != refined.get("pacing"):
            changes.append(f"Pacing changed: {original.get('pacing')} → {refined.get('pacing')}")
        
        # Compare timeline length
        orig_timeline = original.get("timeline", {}).get("clips", [])
        refined_timeline = refined.get("timeline", {}).get("clips", [])
        
        if len(orig_timeline) != len(refined_timeline):
            changes.append(f"Timeline clips: {len(orig_timeline)} → {len(refined_timeline)}")
        
        return changes
    
    def _format_media_analysis_for_context(self, analyses: Dict[str, Any]) -> str:
        """Format media analysis for context string"""
        if not analyses:
            return "No media analysis available"
        
        formatted = []
        for filename, analysis in analyses.items():
            formatted.append(f"📄 {filename}:")
            formatted.append(f"   Type: {analysis.get('file_type', 'unknown')}")
            formatted.append(f"   Analysis: {analysis.get('analysis', 'N/A')[:200]}...")
        
        return "\n".join(formatted)
    
    def _format_plans_for_context(self, plans: List[Dict[str, Any]]) -> str:
        """Format editing plans for context string"""
        if not plans:
            return "No previous plans"
        
        formatted = []
        for i, plan in enumerate(plans, 1):
            formatted.append(f"📋 Plan {i} ({plan.get('plan_id', 'unknown')}):")
            formatted.append(f"   Style: {plan.get('style', 'N/A')}")
            formatted.append(f"   Pacing: {plan.get('pacing', 'N/A')}")
            formatted.append(f"   Duration: {plan.get('duration', 'N/A')}")
            
            # Add key timeline info
            timeline = plan.get("timeline", {})
            clips = timeline.get("clips", [])
            if clips:
                formatted.append(f"   Clips: {len(clips)} segments")
        
        return "\n".join(formatted)
    
    def _format_refinements_for_context(self, refinements: List[Dict[str, Any]]) -> str:
        """Format refinement history for context string"""
        if not refinements:
            return "No previous refinements"
        
        formatted = []
        for i, refinement in enumerate(refinements, 1):
            formatted.append(f"🔄 Refinement {i}:")
            formatted.append(f"   Request: {refinement.get('refinement_prompt', 'N/A')}")
            changes = refinement.get('changes_made', [])
            if changes:
                formatted.append(f"   Changes: {', '.join(changes)}")
        
        return "\n".join(formatted)


# =============================================================================
# CLEAN MODULAR ARCHITECTURE - Following Workflow Diagram
# INPUT → MEDIA_ANALYSER → LLM → EXTRACTOR → SHOTSTACK → GUI
# =============================================================================

class MediaAnalyser:
    """Centralized media analysis component - handles all media types"""
    
    def __init__(self):
        self.video_client = None
        self.audio_client = None
        print("📹 MediaAnalyser initialized")
    
    # Public wrappers for direct use by UI or other components
    def analyze_video(self, file_path: str) -> Dict[str, Any]:
        return self._analyze_video(file_path)

    def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        return self._analyze_audio(file_path)

    def analyze_image(self, file_path: str) -> Dict[str, Any]:
        return self._analyze_image(file_path)
    
    def analyze(self, media_files: List[str]) -> Dict[str, Any]:
        """Analyze all media files and return structured data"""
        if not media_files:
            return {}
        
        print(f"🔍 Analyzing {len(media_files)} media files...")
        analyzed_data = {}
        
        for file_path in media_files:
            filename = os.path.basename(file_path)
            file_type = self._get_file_type(file_path)
            
            try:
                if file_type == "video":
                    analysis = self._analyze_video(file_path)
                elif file_type == "audio":
                    analysis = self._analyze_audio(file_path)
                elif file_type == "image":
                    analysis = self._analyze_image(file_path)
                else:
                    analysis = self._get_fallback_analysis(file_path, filename)
                
                analyzed_data[filename] = analysis
                print(f"✅ {filename} analyzed successfully")
                
            except Exception as e:
                print(f"❌ Failed to analyze {filename}: {str(e)}")
                analyzed_data[filename] = self._get_fallback_analysis(file_path, filename, str(e))
        
        return analyzed_data
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
            return "video"
        elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']:
            return "audio"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return "image"
        return "unknown"
    
    def _analyze_video(self, file_path: str) -> Dict[str, Any]:
        """Analyze video using MiniCPM-V-4.5 via akhaliq/MiniCPM-V-4_5-video-chat"""
        try:
            if not self.video_client:
                print(f"🔗 Connecting to video analysis API: {GRADIO_VIDEO_API}")
                self.video_client = Client(GRADIO_VIDEO_API)
                print(f"✅ Video client initialized successfully")
            
            # Professional analysis question
            analysis_question = """Analyze this video for professional editing and provide:
**Visual Content**: Scenes, objects, people, actions, composition
**Technical Quality**: Resolution, lighting, camera movement, stability  
**Emotional Tone**: Mood, atmosphere, energy level
**Audio Elements**: Music, dialogue, sound effects
**Editing Opportunities**: Suggested cuts, transitions, effects
**Story Flow**: Narrative structure and pacing
**Editing**: Suggest best editing transtion and effectspoints with timestamps in seconds,
Provide detailed analysis for video editing purposes with best editing suggestions"""

            video_file = handle_file(file_path)
            
            # Use new API format
            result = self.video_client.predict(
                video={"video": video_file},
                question=analysis_question,
                fps=VIDEO_ANALYSIS_FPS,
                force_packing=VIDEO_ANALYSIS_FORCE_PACKING,
                history=[],
                api_name=VIDEO_ANALYSIS_PRIMARY_ENDPOINT
            )
            
            return {
                "file_type": "video",
                "analysis": str(result),
                "metadata": {
                    "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                    "analysis_method": "MiniCPM-V-4.5-video-chat",
                    "fps_used": 3
                },
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Video analysis failed: {str(e)}")
            
            # Try alternative endpoint if main fails
            if "api_name" in str(e).lower() or "parameter" in str(e).lower():
                try:
                    print("🔄 Trying alternative API endpoint...")
                    result = self.video_client.predict(
                        video={"video": handle_file(file_path)},
                        question=analysis_question,
                        fps=VIDEO_ANALYSIS_FPS,
                        force_packing=VIDEO_ANALYSIS_FORCE_PACKING,
                        history=[],
                        api_name=VIDEO_ANALYSIS_FALLBACK_ENDPOINT
                    )
                    
                    return {
                        "file_type": "video",
                        "analysis": str(result),
                        "metadata": {
                            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                            "analysis_method": "MiniCPM-V-4.5-video-chat-alt"
                        },
                        "status": "success"
                    }
                except Exception as alt_e:
                    print(f"❌ Alternative endpoint also failed: {str(alt_e)}")
            
            # Reset client for next attempt
            self.video_client = None
            return {
                "file_type": "video",
                "analysis": f"Video analysis failed: {str(e)}. Using fallback analysis.",
                "metadata": {"file_size_mb": os.path.getsize(file_path) / (1024 * 1024)},
                "status": "fallback"
            }
        
        except Exception as e:
            # Specific handling for GPU aborts: wait 3s and retry with lower FPS and fallback endpoint
            if "gpu task aborted" in str(e).lower():
                try:
                    print("⏳ GPU aborted, waiting 3s and retrying with lower FPS on fallback endpoint...")
                    time.sleep(3)
                    result = self.video_client.predict(
                        video={"video": handle_file(file_path)},
                        question=analysis_question,
                        fps=1,
                        force_packing=VIDEO_ANALYSIS_FORCE_PACKING,
                        history=[],
                        api_name=VIDEO_ANALYSIS_FALLBACK_ENDPOINT
                    )
                    return {
                        "file_type": "video",
                        "analysis": str(result),
                        "metadata": {
                            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                            "analysis_method": "MiniCPM-V-4.5-video-chat-retry",
                            "fps_used": 1
                        },
                        "status": "success"
                    }
                except Exception as retry_e:
                    print(f"❌ Retry after GPU abort failed: {retry_e}")
            # Fall through to existing alternative handling
            print(f"❌ Video analysis failed: {str(e)}")
            
            # Try alternative endpoint if main fails
            if "api_name" in str(e).lower() or "parameter" in str(e).lower():
                try:
                    print("🔄 Trying alternative API endpoint...")
                    result = self.video_client.predict(
                        video={"video": handle_file(file_path)},
                        question=analysis_question,
                        fps=VIDEO_ANALYSIS_FPS,
                        force_packing=VIDEO_ANALYSIS_FORCE_PACKING,
                        history=[],
                        api_name=VIDEO_ANALYSIS_FALLBACK_ENDPOINT
                    )
                    
                    return {
                        "file_type": "video",
                        "analysis": str(result),
                        "metadata": {
                            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                            "analysis_method": "MiniCPM-V-4.5-video-chat-alt"
                        },
                        "status": "success"
                    }
                except Exception as alt_e:
                    print(f"❌ Alternative endpoint also failed: {str(alt_e)}")
            
            # Reset client for next attempt
            self.video_client = None
            return {
                "file_type": "video",
                "analysis": f"Video analysis failed: {str(e)}. Using fallback analysis.",
                "metadata": {"file_size_mb": os.path.getsize(file_path) / (1024 * 1024)},
                "status": "fallback"
            }
    
    def _analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio using MidasHeng LM-7B with optimized connection handling"""
        try:
            if not self.audio_client:
                print(f"🔗 Connecting to audio analysis API: {GRADIO_AUDIO_API}")
                
                # Reduce logging to minimize console output
                import logging
                gradio_logger = logging.getLogger("gradio_client")
                httpx_logger = logging.getLogger("httpx")
                original_gradio_level = gradio_logger.level
                original_httpx_level = httpx_logger.level
                
                gradio_logger.setLevel(logging.ERROR)
                httpx_logger.setLevel(logging.ERROR)
                
                self.audio_client = Client(GRADIO_AUDIO_API)
                
                # Restore original logging levels
                gradio_logger.setLevel(original_gradio_level)
                httpx_logger.setLevel(original_httpx_level)
                
                print(f"✅ Audio client initialized successfully")
            
            # Optimized prompt for faster processing
            prompt = """
    Analyze this audio and return JSON with:
    - basic_info (genre, bpm, key, time_signature)
    - structure (sections + timestamps)
    - rhythm (beat grid, downbeats)
    - harmony (chords, melody)
    - instrumentation (instruments/layers per section)
    - dynamics (energy curve, loudness, peaks)
    - mood (emotion, use case)
    - technical (spectrum, stereo width, issues)
    - editing (cut points, drops, fades)
    Suggest best editing transtion and effectspoints with timestamps in seconds,
    clean JSON only.
    """,
            audio_file = handle_file(file_path)
            result = self.audio_client.predict(prompt, audio_file, api_name="/infer")
            
            return {
                "file_type": "audio",
                "analysis": str(result),
                "metadata": {
                    "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                    "analysis_method": "MidasHeng-LM-7B-Optimized"
                },
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Audio analysis failed: {str(e)}")
            # Don't reset client immediately unless it's a connection error
            if "connection" in str(e).lower() or "config" in str(e).lower():
                self.audio_client = None
            return {
                "file_type": "audio",
                "analysis": f"Audio analysis failed: {str(e)}. Using fallback analysis.",
                "metadata": {"file_size_mb": os.path.getsize(file_path) / (1024 * 1024)},
                "status": "fallback"
            }
    
    def _analyze_image(self, file_path: str) -> Dict[str, Any]:
        """Analyze image with basic metadata"""
        file_size = os.path.getsize(file_path)
        return {
            "file_type": "image",
            "analysis": f"Image file suitable for graphics and overlays. File size: {file_size/1024/1024:.1f}MB",
            "metadata": {"file_size_mb": file_size / (1024 * 1024)},
            "status": "success"
        }
    
    def _get_fallback_analysis(self, file_path: str, filename: str, error: str = "") -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        return {
            "file_type": self._get_file_type(file_path),
            "analysis": f"Basic analysis for {filename}. {error}",
            "metadata": {"file_size_mb": os.path.getsize(file_path) / (1024 * 1024)},
            "status": "fallback"
        }


class DirectorAgent:
    """Director Agent - Creates semantic editing plans and creative content"""
    
    def __init__(self):
        self.request_count = 0
        self._client = None
        self.available = False
        
        # Initialize Gemini client
        try:
            if GOOGLE_API_KEY:
                from google import genai
                import os as _os
                if not _os.environ.get("GOOGLE_API_KEY"):
                    _os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                self._client = genai.Client(api_key=GOOGLE_API_KEY)
                self.available = True
                print("🎭 DirectorAgent initialized with Gemini API - semantic planning specialist")
            else:
                print("⚠️ DirectorAgent: GOOGLE_API_KEY not set; will use fallback")
        except Exception as e:
            print(f"⚠️ DirectorAgent Gemini init failed: {e}; will use fallback")
            self.available = False
        
        # Director system prompt focused on producing a human-readable Editing Script (not JSON plan)
        self.DIRECTOR_PROMPT = (
            "You are an award-winning film director and professional video editor AI.\n"
            "Think like a human creative — not a machine.\n"
            "Your job: convert `user_prompt` + `analyzed_media` into a concise, step-by-step **Editing Script** that an editor can directly follow.\n"
            "\n---\n\n"
            "### 🎯 Goal:\n"
            "Produce a compelling editing script (plain language, with approximate timecodes) describing what to cut, when, and why — including pacing, transitions, titles text, and emotional intent.\n"
            "This script will be given to a separate Editor Agent that turns it into Shotstack JSON.\n\n"
            "---\n\n"
            "### 🧠 Inputs you receive:\n"
            "- user_prompt: the user's ask.\n"
            "- analyzed_media: available media with filenames and brief analysis.\n\n"
            "---\n\n"
            "### 🪄 How to write the Editing Script:\n"
            "- Reference ONLY provided filenames. No new media, no URLs.\n"
            "- Use loose time ranges (e.g., 0s–3s, 3s–7s).\n"
            "- Describe visual action, pacing (fast/slow), and desired transition/effect names (from valid set like fade, slideLeft, zoomIn).\n"
            "- Specify titles text and when they appear.\n"
            "- Mention soundtrack usage ONLY if user provided audio.\n"
            "- Keep it brief, clear, and executable.\n\n"
            "---\n\n"
            "### 🧾 Output (JSON with three keys):\n"
            "{\n"
            "  \"content\": \"Short enticing description for the final video\",\n"
            "  \"editing_script\": \"Numbered steps with timecodes for video editing\",\n"
            "  \"music_script\": \"High-level guidance for AI music generation: style, mood, bpm, structure, key moments to accent (e.g., beat drop at 6s)\"\n"
            "}\n\n"
            "Rules: return ONLY this JSON object (no markdown, no prose outside JSON)."
        )
    
    def plan(self, text_prompt: str, analyzed_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Create semantic editing plan and creative content
        
        Args:
            text_prompt: User's creative request
            analyzed_data: Media analysis results
            
        Returns:
            Tuple of (director_content, abstract_plan)
        """
        
        self.request_count += 1
        print(f"🎭 Director Request #{self.request_count}")
        
        # Build a compact context payload for the user message (inputs only)
        context = self._build_director_context(analyzed_data, text_prompt)
        print("🎭 DIRECTOR INPUT | prompt & analyzed_media prepared")
        
        # If Gemini is not available, use fallback
        if not self.available or not self._client:
            print("⚠️ Gemini not available, using fallback...")
            return self._generate_fallback_director_response(analyzed_data, text_prompt)
        
        # Try multiple request strategies with Gemini
        for attempt in range(3):
            try:
                print(f"📡 Director Gemini API attempt {attempt + 1}/3...")
                print("⏳ Waiting for Gemini response...")
                
                # Use Gemini API
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        {"role": "user", "parts": [{"text": f"{self.DIRECTOR_PROMPT}\n\n{context}"}]}
                    ]
                )
                
                result = response.text if hasattr(response, 'text') else str(response)
                
                if result and result.strip():
                    print("✅ Director response received successfully from Gemini")
                    # Log FULL raw response for debugging
                    try:
                        print("=" * 80)
                        print("🎭 DIRECTOR FULL RAW RESPONSE:")
                        print(result)
                        print("=" * 80)
                    except Exception:
                        pass
                    content, abstract_plan = self._parse_director_response(result)
                    # Log parsed summary
                    try:
                        has_editing_script = "editing_script" in (abstract_plan or {})
                        has_music_script = "music_script" in (abstract_plan or {})
                        style = (abstract_plan or {}).get("style")
                        dur = (abstract_plan or {}).get("target_duration")
                        tracks_count = len((abstract_plan or {}).get("tracks", []))
                        print(f"🎭 PARSED PLAN | editing_script={has_editing_script}, music_script={has_music_script}, style={style}, duration={dur}s, tracks={tracks_count}")
                    except Exception:
                        pass
                    return content, abstract_plan
                else:
                    print("⚠️ Empty director response, retrying...")
                    continue
                        
            except Exception as e:
                print(f"❌ Director Gemini error: {str(e)}")
                if attempt < 2:
                    time.sleep(3)
                    continue
        
        # Fallback if all attempts failed
        print("🛠️ Director fallback - generating basic plan...")
        return self._generate_fallback_director_response(analyzed_data, text_prompt)
    
    def _build_director_context(self, analyzed_data: Dict[str, Any], prompt: str) -> str:
        """Build context for director agent"""
        
        # Analyze available media
        media_summary = []
        available_files = []
        
        if analyzed_data:
            for filename, analysis in analyzed_data.items():
                if isinstance(analysis, dict):
                    file_type = analysis.get("file_type", "unknown")
                    analysis_text = analysis.get("analysis", "")
                    
                    # Extract key insights for director
                    media_summary.append(f"📁 **{filename}** ({file_type})")
                    if analysis_text:
                        # Summarize analysis for director (first 200 chars)
                        summary = analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
                        media_summary.append(f"   Analysis: {summary}")
                    
                    available_files.append(filename)
        
        media_context = "\n".join(media_summary) if media_summary else "No media files provided"
        files_list = ", ".join(available_files) if available_files else "none"
        
        context = f"""You are a CREATIVE DIRECTOR for viral video content.

USER REQUEST: {prompt}

AVAILABLE MEDIA:
{media_context}

AVAILABLE FILES: {files_list}

Your task is to return ONLY a JSON object with THREE keys:
1. "content" - Short enticing description for the final video
2. "editing_script" - Numbered steps with timecodes and instructions for video editing
3. "music_script" - High-level guidance for AI music generation

EXAMPLE OUTPUT:
{{
  "content": "A cinematic 15-second edit with dramatic pacing",
  "editing_script": "1) 0-3s: Open with file1.mp4, zoomIn effect, fade in title 'START'. 2) 3-7s: Cut to file2.mp4 with slideLeft transition...",
  "music_script": "Style: cinematic/epic, Mood: dramatic, BPM: 90, Structure: soft intro (0-3s), build (3-7s), energetic peak (7-12s), resolved outro (12-15s). Key moments: gentle swell at 1s, accent at 3s, beat hits at 7s and 10s"
}}

EDITING SCRIPT RULES:
- Reference ONLY provided filenames
- Use loose time ranges (e.g., 0-3s, 3-7s)
- Describe visual action, pacing, transitions, effects, and titles
- Keep brief and executable

MUSIC SCRIPT RULES:
- Specify style, mood, BPM (approximate)
- Describe structure with timing (intro/build/peak/outro)
- Indicate key moments where music should accent the video (e.g., beat drop at 6s)
- Match the emotional arc of the video

Return ONLY the JSON object - no markdown, no explanations."""

        return context
    
    def _parse_director_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse director response.
        Supports legacy {content, abstract_plan}, previous {editing_plan}, and new {content, editing_script}."""
        try:
            import json
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            # New preferred format: {content, editing_script, music_script?}
            if isinstance(data, dict) and "editing_script" in data:
                content = data.get("content") or "Creative video edit planned"
                script = data.get("editing_script") or ""
                # Store script inside abstract_plan for Editor consumption
                abstract_plan = {"editing_script": script}
                try:
                    music_script = data.get("music_script")
                    if isinstance(music_script, str) and music_script.strip():
                        abstract_plan["music_script"] = music_script.strip()
                        print(f"🎵 Extracted music_script: {music_script[:80]}...")
                    else:
                        print("⚠️ No music_script in Director response")
                except Exception as e:
                    print(f"⚠️ Error extracting music_script: {e}")
                return content, abstract_plan
            if "editing_plan" in data and isinstance(data["editing_plan"], dict):
                ep = data["editing_plan"]
                title = ep.get("title") or "Planned Edit"
                theme = ep.get("overall_theme") or "Cinematic"
                tempo = ep.get("tempo") or "dynamic"
                visual_mood = ep.get("visual_mood") or "cinematic"
                ns = ep.get("narrative_structure") or []
                # infer duration from last end_time
                duration = None
                try:
                    if isinstance(ns, list) and ns:
                        duration = int(max((s.get("end_time") or 0) for s in ns if isinstance(s, dict)) or 0)
                except Exception:
                    duration = None
                # Build concise content
                content = f"{title}: {theme}. Tempo: {tempo}."
                # Map narrative beats to key_moments
                key_moments = []
                for s in ns:
                    if not isinstance(s, dict):
                        continue
                    try:
                        tm = max(0, float(s.get("start_time") or 0))
                    except Exception:
                        tm = 0
                    desc = s.get("scene_purpose") or s.get("emotion") or "scene"
                    action = "transition" if s.get("transition") else "cut"
                    key_moments.append({"time": tm, "action": action, "description": desc})
                abstract_plan = {
                    "style": str(visual_mood or "cinematic"),
                    "target_duration": duration or 15,
                    "pacing": str(tempo or "dynamic"),
                    "mood": theme,
                    "hook_strategy": "Follow narrative beats with emotional emphasis",
                    "tracks": [],
                    "key_moments": key_moments,
                    "titles": []
                }
            else:
                content = data.get("content", "Creative video edit planned")
                abstract_plan = data.get("abstract_plan", {})
            
            # CRITICAL FIX: Clean up content if it contains JSON artifacts
            if isinstance(content, str):
                print(f"🔍 DEBUG: Original content: {repr(content)}")
                # Remove markdown code blocks
                content = content.replace("```json", "").replace("```", "").strip()
                print(f"🔍 DEBUG: After removing code blocks: {repr(content)}")
                
                # If content still looks like JSON (starts with { or [), try to extract description
                if content.startswith("{") or content.startswith("["):
                    try:
                        # Try to parse and extract actual content
                        nested_data = json.loads(content)
                        if isinstance(nested_data, dict):
                            print(f"🔍 DEBUG: Parsed nested data keys: {list(nested_data.keys())}")
                            # Extract the actual content description
                            actual_content = nested_data.get("content", "")
                            print(f"🔍 DEBUG: Extracted actual_content: {repr(actual_content)}")
                            if actual_content and not actual_content.startswith("{"):
                                content = actual_content
                            else:
                                # Fallback: generate clean description from abstract_plan
                                style = abstract_plan.get("style", "cinematic")
                                duration = abstract_plan.get("target_duration", 15)
                                mood = abstract_plan.get("mood", "engaging")
                                content = f"A {style} {duration}-second video with {mood} atmosphere."
                    except:
                        # If parsing fails, generate clean description
                        style = abstract_plan.get("style", "cinematic")
                        duration = abstract_plan.get("target_duration", 15)
                        mood = abstract_plan.get("mood", "engaging")
                        content = f"A {style} {duration}-second video with {mood} atmosphere."
                
                # Ensure content is clean and readable (max 500 chars for UI)
                if len(content) > 500:
                    content = content[:497] + "..."
                print(f"🔍 DEBUG: Final cleaned content: {repr(content)}")
            
            if not isinstance(abstract_plan, dict):
                abstract_plan = {}
            abstract_plan.setdefault("style", "cinematic")
            abstract_plan.setdefault("target_duration", 15)
            abstract_plan.setdefault("pacing", "dynamic")
            abstract_plan.setdefault("tracks", [])
            return content, abstract_plan
            
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    content = data.get("content", "Creative video edit planned")
                    abstract_plan = data.get("abstract_plan", {})
                    return content, abstract_plan
                except:
                    pass
            
            # Fallback - treat entire response as content
            return response.strip(), {"style": "cinematic", "target_duration": 15, "tracks": []}
    
    def _generate_fallback_director_response(self, analyzed_data: Dict[str, Any], prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Generate fallback response when director fails"""
        
        # Extract available files
        available_files = []
        video_files = []
        audio_files = []
        
        if analyzed_data:
            for filename, analysis in analyzed_data.items():
                if isinstance(analysis, dict):
                    file_type = analysis.get("file_type", "unknown")
                    available_files.append(filename)
                    if file_type == "video":
                        video_files.append(filename)
                    elif file_type == "audio":
                        audio_files.append(filename)
        
        # Generate basic content
        content = f"Professional video edit incorporating {len(available_files)} media files with dynamic pacing and engaging transitions."
        
        # Generate basic abstract plan
        tracks = []
        if video_files:
            tracks.append({
                "role": "primary_video",
                "media_files": video_files[:1],  # Use first video
                "timing_notes": "main content throughout"
            })
        
        if audio_files:
            tracks.append({
                "role": "background_music",
                "media_files": audio_files[:1],  # Use first audio
                "timing_notes": "background throughout"
            })
        
        tracks.append({
            "role": "titles",
            "media_files": [],
            "timing_notes": "opening and key moments"
        })
        
        abstract_plan = {
            "style": "cinematic",
            "target_duration": 15,
            "pacing": "dynamic",
            "mood": "engaging",
            "hook_strategy": "Strong opening with immediate visual impact",
            "tracks": tracks,
            "key_moments": [
                {"time": 0, "action": "cut", "description": "Strong opening hook"},
                {"time": 7, "action": "transition", "description": "Mid-point transition"},
                {"time": 12, "action": "title_appear", "description": "Closing title"}
            ],
            "titles": [
                {"text": "Watch This", "style_hint": "bold", "timing": "early"},
                {"text": "Amazing", "style_hint": "energetic", "timing": "climax"}
            ]
        }
        
        return content, abstract_plan


class EnhancedLLMProcessor:
    """Enhanced LLM communication component with advanced error handling"""
    
    def __init__(self):
        self.request_count = 0
        self.successful_requests = 0
        self._client = None
        self.available = False
        
        # Initialize Gemini client
        try:
            if GOOGLE_API_KEY:
                from google import genai
                import os as _os
                if not _os.environ.get("GOOGLE_API_KEY"):
                    _os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                self._client = genai.Client(api_key=GOOGLE_API_KEY)
                self.available = True
                print("🤖 EnhancedLLMProcessor initialized with Gemini API")
            else:
                print("⚠️ EnhancedLLMProcessor: GOOGLE_API_KEY not set; will use fallback")
        except Exception as e:
            print(f"⚠️ EnhancedLLMProcessor Gemini init failed: {e}; will use fallback")
            self.available = False
    
    def process(self, analyzed_data: Dict[str, Any], prompt: str) -> str:
        """Process request with LLM using analyzed data and prompt with retry logic"""
        
        self.request_count += 1
        print(f"🤖 LLM Request #{self.request_count}")
        
        # Build enhanced context
        context = self._build_enhanced_context(analyzed_data, prompt)
        
        # DEBUG: Print what we're sending to LLM
        print("=" * 80)
        print("🔍 CONTEXT BEING SENT TO LLM:")
        print(context[:1500] + "..." if len(context) > 1500 else context)
        print("=" * 80)
        
        # If Gemini is not available, use fallback
        if not self.available or not self._client:
            print("⚠️ Gemini not available, using fallback...")
            return self._generate_fallback_llm_response(analyzed_data, prompt)
        
        # Get system prompt
        base_system_prompt = self._get_system_prompt()
        
        # Try multiple request strategies with Gemini
        for attempt in range(3):
            try:
                print(f"📡 LLM Gemini API attempt {attempt + 1}/3...")
                print("⏳ Waiting for Gemini response...")
                
                # Use Gemini API
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        {"role": "user", "parts": [{"text": f"{base_system_prompt}\n\n{context}"}]}
                    ]
                )
                
                result = response.text if hasattr(response, 'text') else str(response)
                
                if result and result.strip():
                    self.successful_requests += 1
                    print("✅ LLM response received successfully from Gemini")
                    return result
                else:
                    print("⚠️ Empty response from Gemini, retrying...")
                    continue
                        
            except Exception as e:
                print(f"❌ LLM Gemini error: {str(e)}")
                if attempt < 2:
                    time.sleep(3)
                    continue
        
        # All attempts failed, generate fallback response
        print("🛠️ All LLM attempts failed, generating fallback response...")
        return self._generate_fallback_llm_response(analyzed_data, prompt)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        
        return """You are a VIRAL VIDEO EDITING GENIUS - a master storyteller who creates content that hooks viewers, builds tension, and goes viral.

🎯 YOUR MISSION: Create timelines that are ADDICTIVE, ENGAGING, and SHAREABLE

=== VIRAL VIDEO PSYCHOLOGY ===
You understand what makes content viral:
✨ HOOK within first 3 seconds (grab attention immediately)
🎭 STORY ARC with tension, conflict, resolution
🎵 RHYTHM that matches music beats and emotional flow
💥 PEAK MOMENTS that create "wow" reactions
🔄 PACING that keeps viewers glued to screen
📱 SOCIAL MEDIA optimization for maximum engagement

=== YOUR EDITING SUPERPOWERS ===
🎬 **Storytelling Master**: You craft narratives that emotionally connect
🎵 **Rhythm Wizard**: You sync cuts to music beats for maximum impact
💡 **Attention Architect**: You design hooks that stop scrolling
🚀 **Viral Engineer**: You create moments that demand sharing
🎨 **Visual Poet**: You use effects and transitions to enhance emotion

You must **only return a single JSON object** in your entire output. 
There must be **no explanations, markdown, or extra text** — only one valid JSON object starting with { and ending with }.

---

=== PERFECT SHOTSTACK JSON EXAMPLE (COPY THIS STRUCTURE EXACTLY) ===

{
  "content": "A dynamic 10-second edit with fast cuts and smooth transitions",
  "json_plan": {
    "timeline": {
      "background": "#000000",
      "tracks": [
        {
          "clips": [
            {
              "asset": {
                "type": "video",
                "src": "https://res.cloudinary.com/example/video.mp4",
                "volume": 1.0,
                "trim": 0.0
              },
              "start": 0.0,
              "length": 3.0,
              "fit": "cover",
              "effect": "zoomIn",
              "filter": "boost",
              "transition": {
                "in": "fade",
                "out": "fadeFast"
              }
            },
            {
              "asset": {
                "type": "video",
                "src": "https://res.cloudinary.com/example/video2.mp4",
                "volume": 1.0,
                "trim": 2.0
              },
              "start": 3.0,
              "length": 4.0,
              "fit": "cover",
              "effect": "slideRight",
              "filter": "contrast",
              "transition": {
                "in": "slideUpFast",
                "out": "fade"
              }
            }
          ]
        },
        {
          "clips": [
            {
              "asset": {
                "type": "title",
                "text": "YOUR MESSAGE HERE",
                "style": "blockbuster",
                "size": "large",
                "color": "#FFFFFF"
              },
              "start": 5.0,
              "length": 2.0,
              "position": "center",
              "transition": {
                "in": "slideUp",
                "out": "fade"
              }
            }
          ]
        },
        {
          "clips": [
            {
              "asset": {
                "type": "audio",
                "src": "https://cdn.shotstack.io/music/cinematic.mp3",
                "volume": 0.6
              },
              "start": 0.0,
              "length": 10.0
            }
          ]
        }
      ]
    },
    "output": {
      "format": "mp4",
      "resolution": "hd",
      "fps": 25
    }
  }
}

⚠️ CRITICAL RULES - FOLLOW EXACTLY:
1. NO COMMENTS in JSON (no // or /* */)
2. ALL numbers must be numeric (not strings): start: 0.0 NOT start: "0.0"
3. ALL strings must use double quotes: "fade" NOT 'fade'
4. NO trailing commas
5. ONLY use properties listed in the rules below

---

=== RULES & VALIDATION ===

1. **JSON FORMAT RULES**
   - You must return ONLY one JSON object.
   - Use double quotes for all strings.
   - No markdown, code fences, or extra text.
   - No trailing commas.
   - The JSON must be parsable directly by the Shotstack API.

2. **TIMELINE STRUCTURE**
   - Required nesting: timeline → tracks → clips → asset
   - Each track contains one or more clips.
   - Each clip must have:
     - "start": number (seconds)
     - "length": number (seconds)
     - "asset": valid Shotstack asset object
   - "length" must always be numeric — never "end" or "auto".
   - "background" must be a hex color (e.g. "#000000").

3. **ASSET RULES** (Properties that go INSIDE the asset object)
   ⚠️ CRITICAL: Assets must ONLY contain asset properties - NEVER nest "timeline", "tracks", "clips", "output" inside an asset!
   
   - type: "video", "image", "audio", or "title" (REQUIRED)
   - src: REQUIRED for video, image, and audio assets (must be full HTTPS URL)
   
   **Title Assets:**
     - "type": "title"
     - "text": required string
     - "style": MUST be one of: "minimal", "blockbuster", "vogue", "sketchy", "skinny", "chunk", "chunkLight", "marker", "future", "subtitle"
       ⚠️ NEVER use: "elegant", "classic", "modern", "subtle", "bold" or any other values!
     - "size": "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"
     - "color": hex color string (e.g. "#ffffff")
       ⚠️ NEVER use: "white", "black", "transparent" - MUST be hex format!
     - "background": hex color string (e.g. "#000000")
       ⚠️ NEVER use: "transparent", "none" - MUST be hex format or omit!
   
   **Video Assets:**
     - "type": "video"
     - "src": HTTPS URL (REQUIRED)
     - "volume": 0.0–1.0 (optional)
     - "trim": start trim time in seconds (optional)
     ⚠️ DO NOT add: "volumeEffect", "speed", "crop" — API rejects these! Note: "crop" is NOT supported anywhere.
   
   **HTML5 Video Assets (for chroma key transitions):**
     - "type": "video" (chromaKey works with standard video type)
     - "src": HTTPS URL (REQUIRED)
     - "volume": 0.0–1.0 (optional)
     - "chromaKey": optional object for green/blue screen removal
       - "color": hex color to remove (e.g. "#00ff00" for green, "#0000ff" for blue)
       - "threshold": 0-255 (default 100, higher = more aggressive removal)
       - "halo": 0-100 (default 50, edge feathering for smooth blending)
   
   **Audio Assets:**
     - "type": "audio"
     - "src": HTTPS URL (REQUIRED)
     - "volume": 0.0–1.0 (optional)
     ⚠️ DO NOT add: "volumeEffect" - API rejects this!
   
   **Image Assets:**
     - "type": "image"
     - "src": HTTPS URL (REQUIRED)

   ALLOWED ASSET KEYS (whitelist):
   - Common: ["type", "src"]
   - Title: ["type", "text", "style", "size", "color", "background"]
   - Video: ["type", "src", "volume", "trim"]
   - Audio: ["type", "src", "volume"]
   - Do NOT add any other keys to asset objects. In particular: no "crop", no "speed", no "volumeEffect", no "loop", no "scale".
   - Video with chromaKey: ["type", "src", "volume", "chromaKey"] - chromaKey has sub-properties: color (hex), threshold (0-255), halo (0-100)

4. **CLIP PROPERTIES** (Properties that go at the CLIP level, NOT inside asset!)
   ⚠️ CRITICAL: These go at the same level as "asset", "start", "length" - NOT inside the asset object!
   
   - "start": number in seconds (REQUIRED)
   - "length": number in seconds (REQUIRED)
   - "asset": asset object (REQUIRED)
   - "fit": "cover" (default), "contain", "crop", or "none"
   - "effect": MUST be one of: "none", "zoomIn", "zoomInSlow", "zoomInFast", "zoomOut", "zoomOutSlow", "zoomOutFast", "slideLeft", "slideLeftSlow", "slideLeftFast", "slideRight", "slideRightSlow", "slideRightFast", "slideUp", "slideUpSlow", "slideUpFast", "slideDown", "slideDownSlow", "slideDownFast"
     ⚠️ NEVER use: "panDown", "pan", "kenBurns" or any other values!
   - "filter": MUST be one of: "none", "boost", "contrast", "darken", "greyscale", "lighten", "muted", "negative"
     ⚠️ NEVER use: "vignette", "blur", "sepia", "highContrast", "lowContrast" or any other values!
   - "position": "center", "top", "bottom", "left", "right", "topLeft", "topRight", "bottomLeft", "bottomRight"
   - "transition": object with "in" and/or "out" keys
     Valid values: "none", fade, fadeSlow, fadeFast, reveal, wipeLeft, wipeRight, slideLeft, slideRight, slideUp, slideDown, carouselLeft, carouselRight, carouselUp, carouselDown, zoom
   - "opacity": 0.0–1.0
   ⚠️ DO NOT add "crop" or "scale" here — both are NOT valid clip properties in this API!

   ALLOWED CLIP KEYS (whitelist):
   - Exactly these keys are allowed on a clip: ["asset", "start", "length", "fit", "effect", "filter", "position", "transition", "opacity"]
   - Do NOT add any other keys at the clip level. In particular: no "speed", no "crop", no "scale", no "transform", no "offset".

5. **OUTPUT RULES (CRITICAL - EXACT FORMAT REQUIRED)**
   - Required properties:
     - "format": "mp4", "gif", "jpg", "png", "bmp", or "mp3" (REQUIRED)
     - "resolution": "preview", "mobile", "sd", "hd", "1080", or "4k" (REQUIRED)
   - Optional properties:
     - "fps": 12, 15, 24, 25, 30, 48, 50, or 60 (OPTIONAL)
     - "aspectRatio": "16:9", "9:16", "1:1", "4:5", or "4:3" (OPTIONAL)
     - "size": {"width": number, "height": number} (OPTIONAL - overrides resolution)
     - "quality": "verylow", "low", "medium", "high", "veryhigh" (OPTIONAL)
     - "scaleTo": "preview", "mobile", "sd", "hd", "1080" (OPTIONAL)
     - "repeat": true/false (OPTIONAL - loop video)
     - "mute": true/false (OPTIONAL - remove audio)

   SOUNDTRACK / AUDIO USAGE RULES:
   - Use a soundtrack ONLY if the user provided at least one audio file in the allowed media list.
   - If you add timeline.soundtrack, its "src" MUST be exactly one of the provided audio URLs. Do NOT invent or reference any external mp3.
   - If the user did not provide audio, DO NOT include a soundtrack object. Prefer leaving audio empty or set output.mute: true.
   - NEVER download or reference stock music or external URLs not listed under allowed media.

⚠️ CRITICAL OUTPUT WARNING:
- Use EITHER "resolution" OR "size" (not both)
- "aspectRatio" works with "resolution" to set dimensions
- Default fps is 25 if not specified

---

=== QUICK REFERENCE: VALID VALUES ONLY ===

**EFFECTS (clip level):**
✅ none, zoomIn, zoomInSlow, zoomInFast, zoomOut, zoomOutSlow, zoomOutFast, slideLeft, slideLeftSlow, slideLeftFast, slideRight, slideRightSlow, slideRightFast, slideUp, slideUpSlow, slideUpFast, slideDown, slideDownSlow, slideDownFast
❌ NEVER: pan, panDown, kenBurns, rotate, spin

**FILTERS (clip level):**
✅ none, boost, contrast, darken, greyscale, lighten, muted, negative

**FORBIDDEN PROPERTIES (anywhere):**
❌ crop, scale, volumeEffect, speed, loop, transcode
Reason: These cause validation errors or are not supported by the current API schema. Prefer using valid clip-level effects (zoomIn/zoomOut, slide*) instead of scale/crop.

**CHROMA KEY (video assets with green/blue screen):**
✅ chromaKey: {"color": "#00ff00", "threshold": 100, "halo": 50}
- Use with "type": "video" assets for green/blue screen removal
- Perfect for transition overlays: place green screen transition video on TOP track, overlaying the cut between two clips
- Common green screen colors: "#00ff00" (green), "#0000ff" (blue), "#000000" (black)
- Threshold: 0-255 (higher = more aggressive), Halo: 0-100 (higher = softer edges)
❌ NEVER: highContrast, lowContrast, vignette, blur, sepia, saturate

**MEDIA USAGE (strict):**
✅ Use ONLY media from the ALLOWED MEDIA ASSETS section (provided filenames/URLs)
❌ NEVER invent new media or external URLs (including mp3 soundtracks) that were not provided

**GENERAL RULE (no invented properties):**
❌ If a property is not listed under ALLOWED CLIP KEYS or ALLOWED ASSET KEYS, DO NOT include it.

**TRANSITIONS (clip level, in/out):**
✅ none, fade, fadeSlow, fadeFast, reveal, revealSlow, revealFast, wipeLeft, wipeRight, slideLeft, slideRight, slideUp, slideDown, carouselLeft, carouselRight, carouselUp, carouselDown, zoom, zoomSlow, zoomFast
❌ NEVER: crossfade, dissolve, cut

**TITLE STYLES (asset level):**
✅ minimal, blockbuster, vogue, sketchy, skinny, chunk, chunkLight, marker, future, subtitle
❌ NEVER: serif, sans, elegant, modern, futuristic, classic, bold

**TITLE SIZES (asset level):**
✅ xx-small, x-small, small, medium, large, x-large, xx-large
❌ NEVER: tiny, huge, or numeric values

**COLORS (asset level):**
✅ Hex format only: "#FFFFFF", "#000000", "#FF0000"
❌ NEVER: "white", "black", "red", "transparent", "none"

**ASSET PROPERTIES - ALLOWED:**
✅ type, src, volume (0.0-1.0), trim (seconds), text, style, size, color, background
❌ NEVER: volumeEffect, speed, crop, loop, transcode, duration
✅ chromaKey: only in video assets - {"color": "#hex", "threshold": 0-255, "halo": 0-100}

---

=== CORRECT STRUCTURE EXAMPLE ===

✅ CORRECT - Properties at the right level:
{
  "clips": [
    {
      "asset": {
        "type": "video",
        "src": "https://example.com/video.mp4",
        "volume": 0.8
      },
      "start": 0,
      "length": 5,
      "fit": "cover",
      "effect": "zoomIn",
      "filter": "boost",
      "position": "center"
    }
  ]
}

❌ WRONG - Don't put clip properties inside asset:
{
  "clips": [
    {
      "asset": {
        "type": "video",
        "src": "https://example.com/video.mp4",
        "crop": {"top": 0.2},  ❌ WRONG! crop is not valid anywhere
        "effect": "zoomIn",    ❌ WRONG! effect goes at clip level
        "filter": "boost"      ❌ WRONG! filter goes at clip level
      },
      "start": 0,
      "length": 5
    }
  ]
}

---

=== BEHAVIOR & REFINEMENT LOGIC ===

- Always build cinematic, clean, and logical edits.
- Time clips in a way that matches the user’s tone or music.
- When the user gives a refinement prompt (e.g. “make it more cinematic”, “add slow motion at the start”), keep the previously generated timeline context and modify only the relevant clips accordingly.
- Ensure every regeneration still outputs a single valid JSON timeline.
- Always return consistent formatting — no deviation from this schema.

---

=== FINAL VALIDATION CHECKLIST (CHECK BEFORE RESPONDING) ===

Before you output your JSON, verify:
✅ 1. NO comments (// or /* */) anywhere in the JSON
✅ 2. ALL property names and values match the QUICK REFERENCE table exactly
✅ 3. ALL colors are in hex format (#FFFFFF, not "white")
✅ 4. ALL numbers are numeric (0.0, not "0.0")
✅ 5. NO forbidden properties anywhere (crop, scale, volumeEffect, speed, loop, transcode)
✅ 5b. chromaKey is ONLY valid in video assets with proper structure: {color: "#hex", threshold: 0-255, halo: 0-100}
✅ 6. ALL transitions/effects/filters are from the valid list
✅ 7. Title styles are from valid list (blockbuster, minimal, etc. - NOT elegant, classic, modern)
✅ 8. Structure is: {content: "...", json_plan: {timeline: {...}, output: {...}}}
✅ 9. NO trailing commas
✅ 10. Valid JSON that can be parsed without errors
✅ 11. "effect", "filter", "fit", "position" are at CLIP level, NOT inside asset
✅ 12. "type", "src", "volume", "text", "style" are INSIDE asset, NOT at clip level
✅ 13. NO "crop" or "scale" property anywhere (not valid in Shotstack API)
✅ 14. NO external media: every asset.src and soundtrack.src must be from the allowed media list
✅ 15. If no audio was provided, timeline.soundtrack MUST be absent (or set output.mute: true)
✅ 16. Clip keys are ONLY from the whitelist [asset,start,length,fit,effect,filter,position,transition,opacity]
✅ 17. Asset keys are ONLY from the corresponding whitelist per asset type

If ANY of these checks fail, FIX IT before responding!

---

=== FINAL REMINDER ===

Remember to always generate 100% correct JSON that can be parsed without errors. Your output must strictly be one valid JSON object matching the schema above — ready for direct use with the Shotstack API.

---

=== EXAMPLE INPUT PROMPTS ===

User: “Create a cinematic trailer edit from this clip: a person running in rain — add dramatic text and emotional music.”

✅ Expected Output:
{
  "content": "High-energy fitness transformation that hooks viewers with dramatic before/after, builds intensity with beat-synced workout clips, and delivers an inspiring climax moment.",
  "json_plan": {
    "timeline": {
      "background": "#000000",
      "tracks": [
        {
          "clips": [
            {
              "asset": {
                "type": "video",
                "src": "https://example.com/workout_start.mp4",
                "volume": 0.8,
                "trim": 0
              },
              "start": 0,
              "length": 2,
              "fit": "cover",
              "transition": {"in": "zoom", "out": "slideRight"},
              "effect": "zoomIn",
              "filter": "boost"
            },
            {
              "asset": {
                "type": "video",
                "src": "https://example.com/workout_intense.mp4",
                "volume": 0.9,
                "trim": 2
              },
              "start": 2,
              "length": 3,
              "fit": "cover",
              "transition": {"in": "slideLeft", "out": "wipeRight"},
              "effect": "slideUp",
              "filter": "contrast"
            },
            {
              "asset": {
                "type": "video",
                "src": "https://example.com/workout_result.mp4",
                "volume": 1,
                "trim": 0
              },
              "start": 5,
              "length": 4,
              "fit": "cover",
              "transition": {"in": "reveal", "out": "fade"},
              "effect": "zoomOut",
              "filter": "boost"
            }
          ]
        },
        {
          "clips": [
            {
              "asset": {
                "type": "title",
                "text": "TRANSFORMATION STARTS NOW",
                "style": "blockbuster",
                "size": "large",
                "color": "#ffffff"
              },
              "start": 0.5,
              "length": 2,
              "position": "center",
              "opacity": 0.95,
              "transition": {"in": "slideUp", "out": "zoom"}
            },
            {
              "asset": {
                "type": "title",
                "text": "NO EXCUSES",
                "style": "blockbuster",
                "size": "x-large",
                "color": "#ff4444"
              },
              "start": 6,
              "length": 2.5,
              "position": "center",
              "opacity": 0.9,
              "transition": {"in": "zoom", "out": "fade"}
            }
          ]
        },
        {
          "clips": [
            {
              "asset": {
                "type": "audio",
                "src": "https://example.com/high_energy_music.mp3",
                "volume": 0.7
              },
              "start": 0,
              "length": 9
            }
          ]
        }
      ]
    },
    "output": {
      "format": "mp4",
      "resolution": "hd",
      "aspectRatio": "9:16",
      "fps": 25
    }
  }
}

---

🚨 MANDATORY VALIDATION RULES:

⚠️ TIMING VALIDATION (CRITICAL):
- Same track clips MUST be sequential: clip2.start = clip1.start + clip1.length
- Different tracks CAN overlap (e.g., video + title overlay)
- NO gaps between clips on same track
- All "start" and "length" values must be positive numbers

⚠️ URL VALIDATION:
- All "src" URLs must be complete HTTPS URLs
- NO placeholder text like "[USE_ACTUAL_URL]"
- URLs must match actual media from context

🚨 FINAL VALIDATION CHECKLIST:
1. ✅ Output contains ONLY: "format", "resolution", "fps"
2. ✅ NO "duration" property in output (causes 400 error)
3. ✅ NO "size" or "aspectRatio" properties
4. ✅ All clips use "length" not "duration"
5. ✅ All assets wrapped in "asset" objects
6. ✅ All tracks inside "timeline" object
7. ✅ NO forbidden properties (volumeEffect, speed, crop)
7b. ✅ chromaKey only in video assets: {"color": "#hex", "threshold": 0-255, "halo": 0-100}
8. ✅ Valid title styles only (blockbuster, vogue, minimal, etc.)
9. ✅ Sequential timing on same track
10. ✅ Real HTTPS URLs, no placeholders
11. ✅ Valid JSON syntax with proper quotes

Your entire output must strictly be one valid JSON object matching the schema above — ready for direct use with the Shotstack API.
"""
    
    def _build_enhanced_context(self, analyzed_data: Dict[str, Any], prompt: str) -> str:
        """Build enhanced context with better structure"""
        
        context_parts = [
            f"🎬 VIDEO EDITING REQUEST: '{prompt}'",
            "",
            "Create a professional video editing plan with detailed JSON timeline.",
        ]

        if analyzed_data:
            context_parts.extend([
                "",
                "📁 ANALYZED MEDIA FILES:",
            ])

            for filename, analysis in analyzed_data.items():
                if isinstance(analysis, dict):
                    file_type = analysis.get("file_type", "unknown")
                    analysis_text = analysis.get("analysis", "No analysis")

                    # Truncate long analysis for better processing
                    if len(analysis_text) > 300:
                        analysis_text = analysis_text[:300] + "..."

                    context_parts.append(f"• {filename} ({file_type.upper()}): {analysis_text}")

            # Allowed media assets list - show cloud URLs directly
            allowed_list = []
            filename_to_url = {}  # Track mapping
            for filename, analysis in analyzed_data.items():
                if isinstance(analysis, dict):
                    # Check cloud_url first, then file_path (which may be cloud URL)
                    cloud_url = analysis.get("cloud_url") or analysis.get("file_path", "")
                    
                    # If it's a Cloudinary URL, use it
                    if cloud_url and cloud_url.startswith("https://res.cloudinary.com"):
                        allowed_list.append(f'  - "{cloud_url}"')
                        filename_to_url[filename] = cloud_url
                    elif cloud_url and cloud_url.startswith("http"):
                        # Any other http URL
                        allowed_list.append(f'  - "{cloud_url}"')
                        filename_to_url[filename] = cloud_url
                    else:
                        # Fallback to filename if no URL
                        allowed_list.append(f'  - "{filename}"')
            
            context_parts.extend([
                "",
                "🔒 ALLOWED MEDIA ASSETS - Use ONLY these URLs in your 'src' fields:",
            ] + allowed_list + [
                "",
                "⚠️ CRITICAL RULES FOR MEDIA:",
                "   1. Copy the FULL HTTPS URL from the list above into 'src' fields",
                "   2. Do NOT use filenames like 'tmpXXX.mp4' or 'placeholder'",
                "   3. Do NOT modify or shorten the URLs",
                "   4. Each asset must use the complete Cloudinary URL shown above",
            ])
        else:
            context_parts.extend([
                "",
                "📝 NOTE: No media files provided - create a conceptual timeline",
            ])

        context_parts.extend([
            "",
            "🎬 PROFESSIONAL VIDEO EDITING PRINCIPLES:",
            "",
            "1. PACING & RHYTHM:",
            "   • Vary clip lengths (2-5s for fast action, 5-10s for story, 3-8s for dialogue)",
            "   • Don't use the same duration for every clip (AVOID repetitive timing!)",
            "   • Match pacing to the content mood (fast=energetic, slow=emotional)",
            "   • Use beat synchronization if music/audio present",
            "",
            "2. VISUAL VARIETY:",
            "   • Mix different effects across clips (don't repeat zoomIn everywhere!)",
            "   • Use effects purposefully: zoomIn (emphasis), zoomOut (reveal), pan (motion)",
            "   • Apply filters selectively (boost=enhance, greyscale=dramatic, contrast=bold)",
            "   • Vary 'fit' property: 'cover' (fill frame), 'contain' (show all), 'crop' (focus)",
            "",
            "3. TRANSITIONS:",
            "   • Use 'fade' for smooth/emotional moments",
            "   • Use 'wipe' or 'slide' for dynamic changes",
            "   • Match transition speed to pacing (Fast/Slow suffixes)",
            "   • Don't overuse - sometimes hard cuts are powerful!",
            "",
            "4. TIMING STRATEGY:",
            "   • START: Strong opening (1-3s) with hook/title",
            "   • MIDDLE: Build momentum with varied segments",
            "   • END: Strong conclusion with fade out",
            "   • Total duration should match user's request or be 15-30s default",
            "",
            "5. MULTI-TRACK COMPOSITION:",
            "   • Track 0: Main video/visual content",
            "   • Track 1: Text overlays (use sparingly, 2-4s duration)",
            "   • Track 2: Additional visuals if needed",
            "   • Text should complement, not overwhelm (appear at key moments)",
            "",
            "6. PROFESSIONAL POLISH:",
            "   • Control audio volume (0.5-1.0) for balance",
            "   • Apply subtle opacity (0.8-0.9) on titles for elegance",
            "   • Position titles strategically ('top'/'bottom' for subtitles, 'center' for main)",
            "   • Add background color to titles for readability",
            "   • Use title styles: 'blockbuster' (bold), 'vogue' (elegant), 'subtitle' (minimal)",
            "",
            "🎯 YOUR TASK:",
            "• Analyze the media content and user request carefully",
            "• Create a VARIED, DYNAMIC timeline (NO repetitive patterns!)",
            "• Think like a professional editor: storytelling, pacing, visual flow",
            "• Use DIFFERENT durations, effects, and transitions across clips",
            "• Only use media URLs from ALLOWED MEDIA ASSETS section",
            "• Make every second count - purposeful, not random",
        ])

        context_parts.extend([
            "",
            "📋 OUTPUT FORMAT:",
            "• Return ONLY valid JSON with 'content' and 'json_plan' keys",
            "• 'content': Brief description of your creative vision",
            "• 'json_plan': Complete Shotstack timeline following ALL rules above",
            "• Include proper tracks, timing, effects, and output settings",
        ])

        return "\n".join(context_parts)
    
    
    def _generate_fallback_llm_response(self, analyzed_data: Dict[str, Any], prompt: str) -> str:
        """Generate a fallback response when LLM fails"""
        
        # Analyze the available data to create intelligent fallback
        has_video = any(data.get("file_type") == "video" for data in analyzed_data.values() if isinstance(data, dict))
        has_audio = any(data.get("file_type") == "audio" for data in analyzed_data.values() if isinstance(data, dict))
        has_image = any(data.get("file_type") == "image" for data in analyzed_data.values() if isinstance(data, dict))
        
        # Create content based on available media
        if has_audio and not has_video:
            content = f"Audio-based video creation for: {prompt}. Using the analyzed audio to create a dynamic visual presentation with waveforms and text overlays."
            tracks = [
                {
                    "type": "audio",
                    "clips": [{
                        "asset": {"type": "audio", "src": "audio_file.mp3", "volume": 0.8},
                        "start": 0,
                        "length": 30
                    }]
                },
                {
                    "type": "title",
                    "clips": [{
                        "asset": {
                            "type": "title",
                            "text": "Audio Visualization",
                            "style": {"fontSize": 60, "color": "#FFFFFF"}
                        },
                        "start": 0,
                        "length": 5
                    }]
                }
            ]
        elif has_video:
            content = f"Video editing plan for: {prompt}. Using the analyzed video content with professional transitions and effects."
            tracks = [
                {
                    "type": "video",
                    "clips": [{
                        "asset": {"type": "video", "src": "video_file.mp4", "volume": 1.0},
                        "start": 0,
                        "length": 30,
                        "transition": {"in": {"type": "fade", "duration": 1}}
                    }]
                }
            ]
        else:
            content = f"Creative video concept for: {prompt}. Generated a basic template that can be customized."
            tracks = [
                {
                    "type": "title",
                    "clips": [{
                        "asset": {
                            "type": "title",
                            "text": "Creative Video",
                            "style": {"fontSize": 80, "color": "#FFFFFF"}
                        },
                        "start": 0,
                        "length": 10
                    }]
                }
            ]
        
        fallback_json = {
            "content": content,
            "json_plan": {
                "timeline": {
                    "background": "#000000",
                    "tracks": tracks
                },
                "output": {
                    "format": "mp4",
                    "resolution": "1920x1080",
                    "fps": 30
                }
            }
        }
        
        return json.dumps(fallback_json, indent=2)
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        success_rate = (self.successful_requests / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.successful_requests,
            "success_rate": f"{success_rate:.1f}%"
        }


class TimelineValidator:
    """Validates and auto-fixes common timeline issues (Quick Fix #2)"""
    
    VALID_TRANSITIONS = {
        "none", "fade", "fadeSlow", "fadeFast", "reveal", "revealSlow", "revealFast",
        "slideLeft", "slideRight", "slideUp", "slideDown",
        "slideLeftSlow", "slideRightSlow", "slideUpSlow", "slideDownSlow",
        "slideLeftFast", "slideRightFast", "slideUpFast", "slideDownFast",
        "wipeLeft", "wipeRight", "wipeLeftSlow", "wipeRightSlow", "wipeLeftFast", "wipeRightFast",
        "zoom", "zoomSlow", "zoomFast", "carouselLeft", "carouselRight"
    }
    
    VALID_EFFECTS = {
        "zoomIn", "zoomOut", "zoomInSlow", "zoomOutSlow", "zoomInFast", "zoomOutFast",
        "slideLeft", "slideRight", "slideUp", "slideDown",
        "slideLeftSlow", "slideRightSlow", "slideUpSlow", "slideDownSlow",
        "slideLeftFast", "slideRightFast", "slideUpFast", "slideDownFast"
    }
    
    VALID_FILTERS = {
        "none", "blur", "boost", "contrast", "darken", "greyscale", 
        "lighten", "muted", "negative"
    }
    
    VALID_TITLE_STYLES = {
        "minimal", "blockbuster", "vogue", "sketchy", "skinny",
        "chunk", "chunkLight", "marker", "future", "subtitle"
    }
    
    @staticmethod
    def validate(timeline: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate timeline and return (is_valid, errors)"""
        errors = []
        
        if not isinstance(timeline, dict):
            return False, ["Timeline must be a dictionary"]
        
        if "timeline" not in timeline:
            errors.append("Missing 'timeline' key")
        if "output" not in timeline:
            errors.append("Missing 'output' key")
        
        if errors:
            return False, errors
        
        tl = timeline.get("timeline", {})
        if "tracks" not in tl or not isinstance(tl["tracks"], list):
            errors.append("Timeline must have 'tracks' array")
            return False, errors
        
        # Validate each track
        for track_idx, track in enumerate(tl["tracks"]):
            if not isinstance(track, dict):
                errors.append(f"Track {track_idx} must be a dictionary")
                continue
            
            clips = track.get("clips", [])
            if not isinstance(clips, list):
                errors.append(f"Track {track_idx} must have 'clips' array")
                continue
            
            # Validate clips
            prev_end = 0
            for clip_idx, clip in enumerate(clips):
                clip_errors = TimelineValidator._validate_clip(
                    clip, track_idx, clip_idx, prev_end
                )
                errors.extend(clip_errors)
                
                start = clip.get("start", 0)
                length = clip.get("length", 0)
                prev_end = start + length
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_clip(clip: Dict[str, Any], track_idx: int, clip_idx: int, prev_end: float) -> List[str]:
        """Validate individual clip"""
        errors = []
        prefix = f"Track {track_idx}, Clip {clip_idx}"
        
        if "asset" not in clip:
            errors.append(f"{prefix}: Missing 'asset'")
            return errors
        
        if "start" not in clip:
            errors.append(f"{prefix}: Missing 'start' time")
        elif not isinstance(clip["start"], (int, float)):
            errors.append(f"{prefix}: 'start' must be numeric")
        
        if "length" not in clip:
            errors.append(f"{prefix}: Missing 'length'")
        elif not isinstance(clip["length"], (int, float)):
            errors.append(f"{prefix}: 'length' must be numeric")
        
        # Check for timing gaps on same track
        if clip_idx > 0:
            start = clip.get("start", 0)
            gap = abs(start - prev_end)
            if gap > 0.1:
                errors.append(f"{prefix}: Timing gap detected ({gap:.2f}s)")
        
        # Validate asset
        asset = clip.get("asset", {})
        asset_type = asset.get("type")
        
        if asset_type in ["video", "audio", "image"]:
            src = asset.get("src", "")
            if not src.startswith("https://"):
                errors.append(f"{prefix}: Invalid URL - must start with https://")
        
        elif asset_type == "title":
            if "text" not in asset:
                errors.append(f"{prefix}: Title missing 'text' field")
            
            style = asset.get("style")
            if style and style not in TimelineValidator.VALID_TITLE_STYLES:
                errors.append(f"{prefix}: Invalid title style '{style}'")
        
        # Validate transition
        transition = clip.get("transition", {})
        if isinstance(transition, dict):
            for key in ["in", "out"]:
                trans = transition.get(key)
                if trans and trans not in TimelineValidator.VALID_TRANSITIONS:
                    errors.append(f"{prefix}: Invalid transition.{key} '{trans}'")
        
        # Validate effect
        effect = clip.get("effect")
        if effect and effect not in TimelineValidator.VALID_EFFECTS:
            errors.append(f"{prefix}: Invalid effect '{effect}'")
        
        # Validate filter
        filter_val = clip.get("filter")
        if filter_val and filter_val not in TimelineValidator.VALID_FILTERS:
            errors.append(f"{prefix}: Invalid filter '{filter_val}'")
        
        return errors
    
    @staticmethod
    def auto_fix(timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to auto-fix common issues"""
        if not isinstance(timeline, dict):
            return timeline
        
        # Ensure basic structure
        if "timeline" not in timeline:
            timeline = {"timeline": timeline, "output": {"format": "mp4", "resolution": "hd", "fps": 25}}
        if "output" not in timeline:
            timeline["output"] = {"format": "mp4", "resolution": "hd", "fps": 25}
        
        tl = timeline.get("timeline", {})
        tracks = tl.get("tracks", [])
        
        # Fix each track
        for track in tracks:
            if not isinstance(track, dict):
                continue
            
            clips = track.get("clips", [])
            prev_end = 0
            
            for clip in clips:
                if not isinstance(clip, dict):
                    continue
                
                # Fix sequential timing
                if "start" in clip and "length" in clip:
                    if clip["start"] > prev_end + 0.1:
                        print(f"🔧 Auto-fixing clip timing: {clip['start']} → {prev_end}")
                        clip["start"] = prev_end
                    prev_end = clip["start"] + clip["length"]
                
                # Fix invalid transitions
                transition = clip.get("transition", {})
                if isinstance(transition, dict):
                    for key in ["in", "out"]:
                        if key in transition:
                            trans = transition[key]
                            if trans not in TimelineValidator.VALID_TRANSITIONS:
                                fixed = TimelineValidator._fix_transition(trans)
                                print(f"🔧 Auto-fixing transition: {trans} → {fixed}")
                                transition[key] = fixed
                
                # Fix invalid effects
                if "effect" in clip:
                    effect = clip["effect"]
                    if effect not in TimelineValidator.VALID_EFFECTS:
                        fixed = TimelineValidator._fix_effect(effect)
                        print(f"🔧 Auto-fixing effect: {effect} → {fixed}")
                        clip["effect"] = fixed
                
                # Fix invalid filters
                if "filter" in clip:
                    filter_val = clip["filter"]
                    if filter_val not in TimelineValidator.VALID_FILTERS:
                        print(f"🔧 Auto-fixing filter: {filter_val} → boost")
                        clip["filter"] = "boost"
                
                # Fix title styles
                asset = clip.get("asset", {})
                if asset.get("type") == "title" and "style" in asset:
                    style = asset["style"]
                    if style not in TimelineValidator.VALID_TITLE_STYLES:
                        print(f"🔧 Auto-fixing title style: {style} → blockbuster")
                        asset["style"] = "blockbuster"
        
        return timeline
    
    @staticmethod
    def _fix_transition(trans: str) -> str:
        """Map invalid transition to valid one"""
        mapping = {
            "slide": "slideLeft",
            "slideIn": "slideLeft",
            "slideOut": "slideRight",
            "wipe": "wipeLeft",
            "zoom": "fade",
            "zoomIn": "fade",
            "zoomOut": "fade"
        }
        return mapping.get(trans, "fade")
    
    @staticmethod
    def _fix_effect(effect: str) -> str:
        """Map invalid effect to valid one"""
        mapping = {
            "zoom": "zoomIn",
            "pan": "slideRight",
            "slide": "slideLeft"
        }
        return mapping.get(effect, "zoomIn")


class TransitionAgent:
    """
    Transition Agent - Intelligently selects chroma key transitions from preset library
    
    Workflow: DirectorAgent → TransitionAgent → EditorAgent
    
    Responsibilities:
    - Analyze director's creative plan (mood, pacing, style)
    - Load transition preset library
    - Match cuts to appropriate transitions using AI
    - Enrich plan with transition specifications (URL, chromaKey params, timing)
    """
    
    def __init__(self, library_path: str = "transition_library.json"):
        self.library_path = library_path
        self.transitions = []
        self.categories = {}
        self._client = None
        self.available = False
        
        # Load transition library
        self._load_library()
        
        # Initialize Gemini for intelligent selection
        try:
            if GOOGLE_API_KEY:
                from google import genai
                import os as _os
                if not _os.environ.get("GOOGLE_API_KEY"):
                    _os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                self._client = genai.Client(api_key=GOOGLE_API_KEY)
                self.available = True
                print(f"🎨 TransitionAgent initialized with {len(self.transitions)} transitions from library")
            else:
                print("⚠️ TransitionAgent: GOOGLE_API_KEY not set; will use rule-based selection")
        except Exception as e:
            print(f"⚠️ TransitionAgent Gemini init failed: {e}; using rule-based fallback")
            self.available = False
    
    def _load_library(self):
        """Load transition presets from JSON library"""
        try:
            library_file = Path(self.library_path)
            if not library_file.exists():
                # Try relative to this file
                library_file = Path(__file__).parent / self.library_path
            
            if library_file.exists():
                with open(library_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.transitions = data.get("transitions", [])
                    self.categories = data.get("categories", {})
                print(f"✅ Loaded {len(self.transitions)} transitions from library")
            else:
                print(f"⚠️ Transition library not found at {self.library_path}")
                self.transitions = []
        except Exception as e:
            print(f"❌ Failed to load transition library: {e}")
            self.transitions = []
    
    def select_transitions(self, director_plan: Dict[str, Any], analyzed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze director's editing_script and select appropriate transitions
        
        Args:
            director_plan: Abstract plan from DirectorAgent with editing_script
            analyzed_data: Media analysis results
            
        Returns:
            Enriched plan with selected transition specifications
        """
        print("🎨 TransitionAgent analyzing editing script and selecting transitions...")
        
        if not self.transitions:
            print("⚠️ No transitions available in library, skipping")
            return director_plan
        
        # Extract editing script (primary input)
        editing_script = director_plan.get("editing_script", "")
        if not editing_script or not editing_script.strip():
            print("⚠️ No editing script provided, cannot select transitions")
            return director_plan
        
        # Extract style/mood for context
        style = director_plan.get("style", "cinematic").lower()
        mood = director_plan.get("mood", "engaging").lower()
        
        # Select transitions using AI (analyzes script to determine count and type)
        if self.available and self._client:
            selected_transitions = self._select_with_ai_from_script(editing_script, style, mood)
        else:
            # Fallback: basic rule-based selection
            print("⚠️ AI not available, using basic fallback")
            selected_transitions = []
        
        # Enrich plan with transition data
        enriched_plan = director_plan.copy()
        enriched_plan["transitions"] = selected_transitions
        
        print(f"✅ Selected {len(selected_transitions)} transitions based on editing script")
        return enriched_plan
    
    def _identify_cut_points(self, tracks: List[Dict], key_moments: List[Dict]) -> List[Dict]:
        """Identify where cuts happen in the timeline"""
        cut_points = []
        
        # Analyze tracks to find clip boundaries
        for track_idx, track in enumerate(tracks):
            if not isinstance(track, dict):
                continue
            
            role = track.get("role", "unknown")
            media_files = track.get("media_files", [])
            timing = track.get("timing_notes", "")
            
            # If track has multiple files, there are cuts between them
            if len(media_files) > 1:
                # Estimate cut timing (this is rough, editor will refine)
                for i in range(len(media_files) - 1):
                    cut_points.append({
                        "track": track_idx,
                        "from_file": media_files[i],
                        "to_file": media_files[i + 1],
                        "index": i,
                        "context": timing
                    })
        
        # Also check key_moments for explicit transitions
        for moment in key_moments:
            if moment.get("action") == "transition":
                cut_points.append({
                    "time": moment.get("time", 0),
                    "description": moment.get("description", ""),
                    "explicit": True
                })
        
        return cut_points
    
    def _select_with_ai_from_script(self, editing_script: str, style: str, mood: str) -> List[Dict]:
        """Use Gemini AI to analyze editing script and select transitions intelligently"""
        print("🤖 Using AI to analyze editing script and select transitions...")
        
        # Prepare simplified library for AI
        library_for_ai = []
        for i, t in enumerate(self.transitions):
            library_for_ai.append({
                "index": i,
                "description": t["description"],
                "duration": t["duration"]
            })
        
        prompt = f"""You are a professional video editor. Analyze the editing script below and select the BEST transitions from the library.

EDITING SCRIPT:
{editing_script}

PROJECT CONTEXT:
- Style: {style}
- Mood: {mood}

AVAILABLE TRANSITIONS (by index):
{json.dumps(library_for_ai, indent=2)}

YOUR TASK:
1. Analyze the editing script to identify where cuts/transitions occur
2. Determine how many transitions are needed (could be 0, 1, 3, 4, or more - based on the script)
3. For each transition point, select the BEST matching transition from the library
4. Match transition style to the content (e.g., fast whoosh for action, soft dissolve for emotional, glitch for tech)
5. Vary transitions for visual interest

OUTPUT FORMAT (return ONLY valid JSON array of integers):
[0, 5, 12]  ← indices of selected transitions in order

If NO transitions are needed, return: []

Be intelligent: analyze the script's pacing, mood, and content to pick transitions that enhance the story."""
        
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.7, "top_p": 0.95}
            )
            
            response_text = (getattr(response, "text", None) or "").strip()
            
            # Parse AI response - expect array of integers
            import re
            json_match = re.search(r'\[[\d\s,]*\]', response_text)
            if json_match:
                selected_indices = json.loads(json_match.group())
                
                # Build selected transitions with full data
                selected = []
                for idx in selected_indices:
                    if isinstance(idx, int) and 0 <= idx < len(self.transitions):
                        transition = self.transitions[idx]
                        selected.append({
                            "transition": transition,
                            "timing_note": f"Use this transition at appropriate cut point"
                        })
                    else:
                        print(f"⚠️ Invalid transition index: {idx}")
                
                print(f"✅ AI selected {len(selected)} transitions from library")
                return selected
            else:
                print("⚠️ AI response not parsable, no transitions selected")
                return []
                
        except Exception as e:
            print(f"❌ AI selection failed: {e}")
            return []
    
    def _select_with_rules(self, cut_points: List[Dict], style: str, pacing: str, mood: str) -> List[Dict]:
        """Rule-based transition selection (fallback when AI unavailable)"""
        print("📏 Using rule-based transition selection...")
        
        selected = []
        
        # Filter transitions by style/mood
        filtered = [t for t in self.transitions 
                   if style in t["categories"] or mood in t["moods"]]
        
        if not filtered:
            # No match, use all
            filtered = self.transitions
        
        # Further filter by pacing
        if pacing == "fast":
            filtered = [t for t in filtered if t["pacing"] in ["fast", "medium"] and t["duration"] <= 1.2]
        elif pacing == "slow":
            filtered = [t for t in filtered if t["pacing"] in ["slow", "medium"] and t["duration"] >= 1.2]
        
        if not filtered:
            filtered = self.transitions
        
        # Assign transitions to cuts (with variety)
        for i, cut in enumerate(cut_points):
            transition = filtered[i % len(filtered)]  # Cycle through options
            selected.append({
                "cut_index": i,
                "transition": transition,
                "cut_point": cut
            })
        
        return selected


class EditorAgent:
    """Editor Agent - Converts abstract plans to Shotstack JSON timelines"""
    
    def __init__(self):
        self.request_count = 0
        self._client = None
        self.available = False
        
        # Initialize Gemini client
        try:
            if GOOGLE_API_KEY:
                from google import genai
                import os as _os
                if not _os.environ.get("GOOGLE_API_KEY"):
                    _os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                self._client = genai.Client(api_key=GOOGLE_API_KEY)
                self.available = True
                print("🎬 EditorAgent initialized with Gemini API - Shotstack JSON specialist")
            else:
                print("⚠️ EditorAgent: GOOGLE_API_KEY not set; will use fallback")
        except Exception as e:
            print(f"⚠️ EditorAgent Gemini init failed: {e}; will use fallback")
            self.available = False
        
        # SIMPLIFIED Editor prompt with concrete examples (Quick Fix #1)
        self.EDITOR_PROMPT = (
            "You are a professional video editor. Convert the editing plan to valid Shotstack JSON.\n\n"
            "🎯 OUTPUT FORMAT: Return ONLY a JSON object (no markdown, no text before/after):\n"
            "{\n"
            "  \"timeline\": {\"background\": \"#000000\", \"tracks\": [...]},\n"
            "  \"output\": {\"format\": \"mp4\", \"resolution\": \"hd\", \"fps\": 25}\n"
            "}\n\n"
            "⚡ CRITICAL RULES:\n"
            "1. JSON only - NO markdown, NO explanations, NO text before { or after }\n"
            "2. All strings use double quotes: \"type\": \"video\"\n"
            "3. All numbers are numeric: \"start\": 0 (not \"0\")\n"
            "4. NO trailing commas\n"
            "5. Use ONLY provided HTTPS URLs from media list\n"
            "6. Sequential timing on same track: clip2.start = clip1.start + clip1.length\n"
            "7. Valid transitions: fade, slideLeft, slideRight, slideUp, slideDown, wipeLeft, wipeRight, reveal (add Fast/Slow for variants)\n"
            "8. Valid effects: zoomIn, zoomOut, slideLeft, slideRight, slideUp, slideDown (add Fast/Slow)\n"
            "9. Valid filters: boost, contrast, blur, greyscale, lighten, darken, muted, negative, none\n"
            "10. Valid title styles: blockbuster, vogue, minimal, subtitle, future, marker, chunk, skinny\n\n"
            "📋 ASSET TYPES:\n"
            "• Video: {\"type\": \"video\", \"src\": \"URL\", \"volume\": 0.8}\n"
            "• Audio: {\"type\": \"audio\", \"src\": \"URL\", \"volume\": 0.6}\n"
            "• Image: {\"type\": \"image\", \"src\": \"URL\"}\n"
            "• Title: {\"type\": \"title\", \"text\": \"TEXT\", \"style\": \"blockbuster\", \"size\": \"large\", \"color\": \"#FFFFFF\"}\n\n"
            "🎬 EXAMPLE 1 - Simple Video + Title:\n"
            "{\n"
            "  \"timeline\": {\n"
            "    \"background\": \"#000000\",\n"
            "    \"tracks\": [\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/video/upload/v1/sample.mp4\", \"volume\": 1},\n"
            "            \"start\": 0,\n"
            "            \"length\": 10,\n"
            "            \"fit\": \"cover\",\n"
            "            \"transition\": {\"in\": \"fade\", \"out\": \"fade\"},\n"
            "            \"effect\": \"zoomIn\"\n"
            "          }\n"
            "        ]\n"
            "      },\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\"type\": \"title\", \"text\": \"WATCH THIS\", \"style\": \"blockbuster\", \"size\": \"large\", \"color\": \"#FFFFFF\"},\n"
            "            \"start\": 1,\n"
            "            \"length\": 3,\n"
            "            \"position\": \"center\",\n"
            "            \"transition\": {\"in\": \"slideUp\"}\n"
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]\n"
            "  },\n"
            "  \"output\": {\"format\": \"mp4\", \"resolution\": \"hd\", \"fps\": 25}\n"
            "}\n\n"
            "🎬 EXAMPLE 2 - Multi-Clip Sequence:\n"
            "{\n"
            "  \"timeline\": {\n"
            "    \"background\": \"#000000\",\n"
            "    \"tracks\": [\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/video/upload/v1/video1.mp4\", \"volume\": 0.8},\n"
            "            \"start\": 0,\n"
            "            \"length\": 5,\n"
            "            \"fit\": \"cover\",\n"
            "            \"transition\": {\"in\": \"fade\", \"out\": \"slideRight\"},\n"
            "            \"filter\": \"boost\"\n"
            "          },\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/video/upload/v1/video2.mp4\", \"volume\": 0.8},\n"
            "            \"start\": 5,\n"
            "            \"length\": 5,\n"
            "            \"fit\": \"cover\",\n"
            "            \"transition\": {\"in\": \"slideLeft\", \"out\": \"fade\"},\n"
            "            \"effect\": \"zoomOut\"\n"
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]\n"
            "  },\n"
            "  \"output\": {\"format\": \"mp4\", \"resolution\": \"hd\", \"fps\": 25}\n"
            "}\n\n"
            "🎬 EXAMPLE 3 - Video + Audio + Multiple Titles:\n"
            "{\n"
            "  \"timeline\": {\n"
            "    \"background\": \"#000000\",\n"
            "    \"soundtrack\": {\"src\": \"https://res.cloudinary.com/demo/video/upload/v1/music.mp3\", \"effect\": \"fadeInFadeOut\", \"volume\": 0.5},\n"
            "    \"tracks\": [\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/video/upload/v1/main.mp4\", \"volume\": 0.3},\n"
            "            \"start\": 0,\n"
            "            \"length\": 15,\n"
            "            \"fit\": \"cover\",\n"
            "            \"effect\": \"zoomInSlow\"\n"
            "          }\n"
            "        ]\n"
            "      },\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\"type\": \"title\", \"text\": \"THE BEGINNING\", \"style\": \"blockbuster\", \"size\": \"x-large\", \"color\": \"#FFFFFF\"},\n"
            "            \"start\": 1,\n"
            "            \"length\": 2.5,\n"
            "            \"position\": \"center\",\n"
            "            \"transition\": {\"in\": \"slideUp\", \"out\": \"fade\"}\n"
            "          },\n"
            "          {\n"
            "            \"asset\": {\"type\": \"title\", \"text\": \"THE JOURNEY\", \"style\": \"vogue\", \"size\": \"large\", \"color\": \"#FFD700\"},\n"
            "            \"start\": 7,\n"
            "            \"length\": 2.5,\n"
            "            \"position\": \"center\",\n"
            "            \"transition\": {\"in\": \"slideLeft\", \"out\": \"slideRight\"}\n"
            "          },\n"
            "          {\n"
            "            \"asset\": {\"type\": \"title\", \"text\": \"THE END\", \"style\": \"blockbuster\", \"size\": \"x-large\", \"color\": \"#FFFFFF\"},\n"
            "            \"start\": 12,\n"
            "            \"length\": 2.5,\n"
            "            \"position\": \"center\",\n"
            "            \"transition\": {\"in\": \"slideDown\"}\n"
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]\n"
            "  },\n"
            "  \"output\": {\"format\": \"mp4\", \"resolution\": \"hd\", \"fps\": 25}\n"
            "}\n\n"
            "🎬 EXAMPLE 4 - With Chroma Key Transition Overlays:\n"
            "{\n"
            "  \"timeline\": {\n"
            "    \"background\": \"#000000\",\n"
            "    \"tracks\": [\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\n"
            "              \"type\": \"video\",\n"
            "              \"src\": \"https://res.cloudinary.com/demo/transitions/orange-circle.mp4\",\n"
            "              \"chromaKey\": {\"color\": \"#000000\", \"threshold\": 100, \"halo\": 50}\n"
            "            },\n"
            "            \"start\": 4,\n"
            "            \"length\": 2\n"
            "          },\n"
            "          {\n"
            "            \"asset\": {\n"
            "              \"type\": \"video\",\n"
            "              \"src\": \"https://res.cloudinary.com/demo/transitions/blue-cloud.mp4\",\n"
            "              \"chromaKey\": {\"color\": \"#ffffff\", \"threshold\": 100, \"halo\": 50}\n"
            "            },\n"
            "            \"start\": 9,\n"
            "            \"length\": 2\n"
            "          }\n"
            "        ]\n"
            "      },\n"
            "      {\n"
            "        \"clips\": [\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/clip1.mp4\", \"volume\": 0.8},\n"
            "            \"start\": 0,\n"
            "            \"length\": 5,\n"
            "            \"fit\": \"cover\"\n"
            "          },\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/clip2.mp4\", \"volume\": 0.8},\n"
            "            \"start\": 5,\n"
            "            \"length\": 5,\n"
            "            \"fit\": \"cover\"\n"
            "          },\n"
            "          {\n"
            "            \"asset\": {\"type\": \"video\", \"src\": \"https://res.cloudinary.com/demo/clip3.mp4\", \"volume\": 0.8},\n"
            "            \"start\": 10,\n"
            "            \"length\": 5,\n"
            "            \"fit\": \"cover\"\n"
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]\n"
            "  },\n"
            "  \"output\": {\"format\": \"mp4\", \"resolution\": \"hd\", \"fps\": 25}\n"
            "}\n\n"
            "Now generate a timeline following these patterns. Match the quality and structure of the examples above.\n"
        )
    
    def build_timeline(self, abstract_plan: Dict[str, Any], analyzed_data: Dict[str, Any], url_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Convert abstract plan to Shotstack-compatible JSON timeline
        
        Args:
            abstract_plan: Semantic plan from DirectorAgent
            analyzed_data: Media analysis results
            url_mappings: Optional filename to URL mappings
            
        Returns:
            Shotstack-compatible JSON timeline
        """
        
        self.request_count += 1
        print(f"🎬 Editor Request #{self.request_count}")
        
        # Build editor context
        context = self._build_editor_context(abstract_plan, analyzed_data, url_mappings)
        
        print("=" * 80)
        print("🎬 EDITOR CONTEXT:")
        print(context[:1000] + "..." if len(context) > 1000 else context)
        print("=" * 80)
        
        # If Gemini is not available, use fallback
        if not self.available or not self._client:
            print("⚠️ Gemini not available, using fallback...")
            return self._generate_fallback_timeline(abstract_plan, analyzed_data, url_mappings)
        
        # Try multiple request strategies with Gemini
        for attempt in range(3):
            try:
                print(f"📡 Editor Gemini API attempt {attempt + 1}/3...")
                print("⏳ Waiting for Gemini response...")
                
                # Use Gemini API
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        {"role": "user", "parts": [{"text": f"{self.EDITOR_PROMPT}\n\n{context}"}]}
                    ]
                )
                
                result = response.text if hasattr(response, 'text') else str(response)
                
                if result and result.strip():
                    print("✅ Editor response received successfully from Gemini")
                    # Log FULL raw response for debugging
                    try:
                        print("=" * 80)
                        print("🎬 EDITOR FULL RAW RESPONSE:")
                        print(result)
                        print("=" * 80)
                    except Exception:
                        pass
                    return self._parse_editor_response(result, analyzed_data, url_mappings)
                else:
                    print("⚠️ Empty editor response, retrying...")
                    continue
                        
            except Exception as e:
                print(f"❌ Editor Gemini error: {str(e)}")
                if attempt < 2:
                    time.sleep(3)
                    continue
        
        # Fallback if all attempts failed
        print("🛠️ Editor fallback - generating basic timeline...")
        return self._generate_fallback_timeline(abstract_plan, analyzed_data, url_mappings)
    
    def _build_editor_context(self, abstract_plan: Dict[str, Any], analyzed_data: Dict[str, Any], url_mappings: Optional[Dict[str, str]]) -> str:
        """Build context for editor agent"""
        
        # Build media URL mapping
        media_urls = {}
        if analyzed_data:
            for filename, analysis in analyzed_data.items():
                if isinstance(analysis, dict):
                    cloud_url = analysis.get("cloud_url") or analysis.get("file_path", "")
                    if cloud_url and cloud_url.startswith("https://"):
                        media_urls[filename] = cloud_url
        
        # Add from url_mappings if provided
        if url_mappings:
            media_urls.update(url_mappings)
        
        # Optional: Director's Editing Script (preferred guidance)
        editing_script = None
        try:
            if isinstance(abstract_plan, dict):
                es = abstract_plan.get('editing_script')
                if isinstance(es, str) and es.strip():
                    editing_script = es.strip()
        except Exception:
            editing_script = None

        # Format abstract plan for context (fallback/augmentation to script)
        plan_summary = []
        plan_summary.append(f"Style: {abstract_plan.get('style', 'cinematic')}")
        plan_summary.append(f"Duration: {abstract_plan.get('target_duration', 15)} seconds")
        plan_summary.append(f"Pacing: {abstract_plan.get('pacing', 'dynamic')}")
        plan_summary.append(f"Mood: {abstract_plan.get('mood', 'engaging')}")
        
        # Format tracks
        tracks_info = []
        for track in abstract_plan.get('tracks', []):
            role = track.get('role', 'unknown')
            files = track.get('media_files', [])
            timing = track.get('timing_notes', '')
            tracks_info.append(f"- {role}: {', '.join(files)} ({timing})")
        
        # Format key moments
        moments_info = []
        for moment in abstract_plan.get('key_moments', []):
            time_val = moment.get('time', 0)
            action = moment.get('action', 'cut')
            desc = moment.get('description', '')
            moments_info.append(f"- {time_val}s: {action} - {desc}")
        
        # Format titles
        titles_info = []
        for title in abstract_plan.get('titles', []):
            text = title.get('text', '')
            style = title.get('style_hint', 'bold')
            timing = title.get('timing', 'mid')
            titles_info.append(f"- '{text}' ({style}, {timing})")
        
        # Format chroma key transitions (from TransitionAgent)
        transitions_info = []
        selected_transitions = abstract_plan.get('transitions', [])
        if selected_transitions:
            for i, trans_data in enumerate(selected_transitions):
                transition = trans_data.get('transition', {})
                transitions_info.append(
                    f"- Transition {i+1}:\n"
                    f"  Description: {transition.get('description', 'N/A')}\n"
                    f"  Duration: {transition.get('duration', 1.0)}s\n"
                    f"  URL: {transition.get('url', 'N/A')}\n"
                    f"  ChromaKey: {json.dumps(transition.get('chromaKey', {}))}"
                )
        
        context = f"""You are a SHOTSTACK JSON EXPERT. Convert the provided Editing Script and/or abstract plan into a valid Shotstack timeline.

EDITING SCRIPT:
{(editing_script if editing_script else 'No explicit script provided; rely on abstract plan below')}

ABSTRACT PLAN:
{chr(10).join(plan_summary)}

TRACKS:
{chr(10).join(tracks_info) if tracks_info else "No specific tracks defined"}

KEY MOMENTS:
{chr(10).join(moments_info) if moments_info else "No specific moments defined"}

TITLES:
{chr(10).join(titles_info) if titles_info else "No titles defined"}

CHROMA KEY TRANSITIONS (use these for smooth transitions between clips):
{chr(10).join(transitions_info) if transitions_info else "No transitions selected - use standard Shotstack transitions"}

AVAILABLE MEDIA URLS:
{chr(10).join([f"- {filename}: {url}" for filename, url in media_urls.items()]) if media_urls else "No media URLs available"}

⚠️ IMPORTANT: If CHROMA KEY TRANSITIONS are provided above, you MUST use them!
- Place each transition video on TRACK 1 (top layer)
- The transition should OVERLAY the cut between two clips
- Timing: start 1-2s before the cut, end 1-2s after the cut
- Use the exact URL and chromaKey settings provided
- Example: if clipA ends at 10s and clipB starts at 10s, place transition from 9s to 12s on track 1

TRACK LAYERING (CRITICAL):
- Track 1 = TOP layer (transition overlays go here)
- Track 2 = Main video clips
- Track 3+ = Additional clips/audio
Remember: Lower track numbers render ON TOP of higher track numbers!

Your task is to return ONLY a valid Shotstack JSON timeline object with this EXACT structure:

{{
  "timeline": {{
    "background": "#000000",
    "tracks": [
      {{
        "clips": [
          {{
            "asset": {{
              "type": "video|audio|image|title",
              "src": "https://full-url-here",
              "volume": 0.8
            }},
            "start": 0,
            "length": 15,
            "fit": "cover",
            "transition": {{"in": "fade", "out": "fade"}},
            "effect": "zoomIn",
            "filter": "boost"
          }}
        ]
      }}
    ]
  }},
  "output": {{
    "format": "mp4",
    "resolution": "hd",
    "fps": 25
  }}
}}

CRITICAL RULES:
1. Use ONLY the provided HTTPS URLs from AVAILABLE MEDIA URLS
2. NO placeholder URLs like "example.com" or "earth.mp4"
3. All clips must have "start" and "length" (never "duration")
4. Sequential timing: clip2.start = clip1.start + clip1.length (same track)
5. Output ONLY contains: format, resolution, fps (NO duration, size, aspectRatio)
6. All asset properties go in "asset" object
7. All clip properties (fit, effect, filter, transition) go on clip level
8. Title assets use: type:"title", text:"text", style:"blockbuster"
9. VALID TRANSITIONS: "none", "fade", "fadeSlow", "fadeFast", "reveal", "revealSlow", "revealFast", "wipeLeft", "wipeLeftSlow", "wipeLeftFast", "wipeRight", "wipeRightSlow", "wipeRightFast", "slideLeft", "slideLeftSlow", "slideLeftFast", "slideRight", "slideRightSlow", "slideRightFast", "slideUp", "slideUpSlow", "slideUpFast", "slideDown", "slideDownSlow", "slideDownFast"
10. CHROMA KEY FORMAT (for transition overlays):
    {{
      "asset": {{
        "type": "video",
        "src": "https://transition-url.mp4",
        "chromaKey": {{"color": "#00ff00", "threshold": 100, "halo": 50}}
      }},
      "start": 9,
      "length": 3
    }}
    IMPORTANT: chromaKey color can be ANY hex color:
    - "#00ff00" for green screen
    - "#0000ff" for blue screen
    - "#000000" for black background
    - "#ffffff" for white background
    Use the EXACT color provided in the transition specs!

Return ONLY the JSON object - no markdown, no explanations."""

        return context
    
    def _strip_json_comments(self, json_str: str) -> str:
        """Remove JavaScript-style comments from JSON string"""
        import re
        # Remove single-line comments (// ...)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        # Remove multi-line comments (/* ... */)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        return json_str
    
    def _parse_editor_response(self, response: str, analyzed_data: Dict[str, Any], url_mappings: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Parse editor response into Shotstack JSON"""
        
        try:
            # Try to parse as JSON directly
            import json
            
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # CRITICAL: Strip comments before parsing (LLM often adds helpful comments)
            cleaned = self._strip_json_comments(cleaned)
            
            data = json.loads(cleaned)
            
            # Validate basic structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Extract timeline from new format: {content: "...", json_plan: {timeline: {...}, output: {...}}}
            if "json_plan" in data:
                timeline = data["json_plan"]
                print(f"✅ Extracted json_plan from response (content: {data.get('content', '')[:50]}...)")
            else:
                timeline = data
            
            # Ensure required structure
            if "timeline" not in timeline:
                timeline = {"timeline": timeline, "output": {"format": "mp4", "resolution": "hd", "fps": 25}}
            
            if "output" not in timeline:
                timeline["output"] = {"format": "mp4", "resolution": "hd", "fps": 25}
            
            # Validate and fix URLs
            timeline = self._ensure_valid_urls(timeline, analyzed_data, url_mappings)
            
            # Validate and fix transitions
            timeline = self._fix_invalid_transitions(timeline)
            
            # APPLY TIMELINEVALIDATOR (Quick Fix #2)
            is_valid, errors = TimelineValidator.validate(timeline)
            if not is_valid:
                print(f"⚠️ Timeline validation found {len(errors)} issue(s):")
                for error in errors[:5]:  # Show first 5
                    print(f"   • {error}")
                
                # Attempt auto-fix
                print("🔧 Attempting auto-fix...")
                timeline = TimelineValidator.auto_fix(timeline)
                
                # Re-validate
                is_valid, remaining_errors = TimelineValidator.validate(timeline)
                if is_valid:
                    print("✅ Auto-fix successful! Timeline is now valid.")
                else:
                    print(f"⚠️ {len(remaining_errors)} issue(s) remain after auto-fix")
            else:
                print("✅ Timeline validation passed!")
            
            return timeline
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"   Attempting to fix and retry...")
            
            # Try multiple fix strategies
            try:
                # Strategy 1: Fix common JSON issues
                fixed_json = cleaned
                # Fix unescaped control characters
                fixed_json = fixed_json.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                # Try parsing again
                data = json.loads(fixed_json)
                
                if "json_plan" in data:
                    timeline = data["json_plan"]
                    print(f"✅ Fixed JSON parse error - extracted json_plan")
                else:
                    timeline = data
                
                if "timeline" not in timeline:
                    timeline = {"timeline": timeline, "output": {"format": "mp4", "resolution": "hd", "fps": 25}}
                if "output" not in timeline:
                    timeline["output"] = {"format": "mp4", "resolution": "hd", "fps": 25}
                
                timeline = self._ensure_valid_urls(timeline, analyzed_data, url_mappings)
                timeline = self._fix_invalid_transitions(timeline)
                return timeline
                
            except Exception as e2:
                print(f"⚠️ Could not fix JSON: {e2}")
                # Last resort: try regex extraction
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        timeline = json.loads(json_match.group())
                        return self._ensure_valid_urls(timeline, analyzed_data, url_mappings)
                    except:
                        pass
                
                # Return fallback only as last resort
                print("⚠️ Using fallback timeline")
                return self._generate_fallback_timeline({}, analyzed_data, url_mappings)
        
        except Exception as e:
            print(f"⚠️ Editor parse error: {e}")
            return self._generate_fallback_timeline({}, analyzed_data, url_mappings)
    
    def _ensure_valid_urls(self, timeline: Dict[str, Any], analyzed_data: Dict[str, Any], url_mappings: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Ensure all asset URLs are valid cloud URLs"""
        
        # Build available URLs
        valid_urls = []
        if analyzed_data:
            for analysis in analyzed_data.values():
                if isinstance(analysis, dict):
                    cloud_url = analysis.get("cloud_url") or analysis.get("file_path", "")
                    if cloud_url and cloud_url.startswith("https://"):
                        valid_urls.append(cloud_url)
        
        if url_mappings:
            for url in url_mappings.values():
                if url and url.startswith("https://") and url not in valid_urls:
                    valid_urls.append(url)
        
        if not valid_urls:
            print("⚠️ No valid URLs available for replacement")
            return timeline
        
        # Replace invalid URLs in timeline
        try:
            tracks = timeline.get("timeline", {}).get("tracks", [])
            url_index = 0
            
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                clips = track.get("clips", [])
                for clip in clips:
                    if not isinstance(clip, dict):
                        continue
                    asset = clip.get("asset", {})
                    if isinstance(asset, dict):
                        src = asset.get("src", "")
                        asset_type = asset.get("type", "")
                        
                        # Check if URL needs replacement
                        if asset_type in ["video", "audio", "image"] and src:
                            if not src.startswith("https://res.cloudinary.com"):
                                # Replace with valid URL
                                replacement_url = valid_urls[url_index % len(valid_urls)]
                                asset["src"] = replacement_url
                                url_index += 1
                                print(f"🔄 Replaced invalid URL with: {replacement_url[:60]}...")
            
            return timeline
            
        except Exception as e:
            print(f"⚠️ URL replacement error: {e}")
            return timeline
    
    def _fix_invalid_transitions(self, timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Fix invalid transition values to valid Shotstack transitions"""
        
        # Valid Shotstack transitions
        valid_transitions = {
            "none", "fade", "fadeSlow", "fadeFast", "reveal", "revealSlow", "revealFast",
            "wipeLeft", "wipeLeftSlow", "wipeLeftFast", "wipeRight", "wipeRightSlow", "wipeRightFast",
            "slideLeft", "slideLeftSlow", "slideLeftFast", "slideRight", "slideRightSlow", "slideRightFast",
            "slideUp", "slideUpSlow", "slideUpFast", "slideDown", "slideDownSlow", "slideDownFast"
        }
        
        # Mapping of common invalid transitions to valid ones
        transition_fixes = {
            "slide": "slideLeft",
            "slideIn": "slideLeft", 
            "slideOut": "slideRight",
            "wipe": "wipeLeft",
            "zoom": "fade",
            "zoomIn": "fade",
            "zoomOut": "fade"
        }
        
        try:
            tracks = timeline.get("timeline", {}).get("tracks", [])
            
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                clips = track.get("clips", [])
                for clip in clips:
                    if not isinstance(clip, dict):
                        continue
                    
                    transition = clip.get("transition", {})
                    if isinstance(transition, dict):
                        # Fix 'in' transition
                        in_trans = transition.get("in")
                        if in_trans and in_trans not in valid_transitions:
                            fixed_trans = transition_fixes.get(in_trans, "fade")
                            transition["in"] = fixed_trans
                            print(f"🔧 Fixed invalid transition 'in': '{in_trans}' → '{fixed_trans}'")
                        
                        # Fix 'out' transition  
                        out_trans = transition.get("out")
                        if out_trans and out_trans not in valid_transitions:
                            fixed_trans = transition_fixes.get(out_trans, "fade")
                            transition["out"] = fixed_trans
                            print(f"🔧 Fixed invalid transition 'out': '{out_trans}' → '{fixed_trans}'")
            
            return timeline
            
        except Exception as e:
            print(f"⚠️ Transition fix error: {e}")
            return timeline
    
    def _generate_fallback_timeline(self, abstract_plan: Dict[str, Any], analyzed_data: Dict[str, Any], url_mappings: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Generate professional fallback timeline when editor fails (Quick Fix #3)"""
        
        print("🛠️ Generating professional fallback timeline...")
        
        # Extract style preferences from abstract plan
        style = abstract_plan.get("style", "cinematic")
        duration = abstract_plan.get("target_duration", 15)
        pacing = abstract_plan.get("pacing", "dynamic")
        
        # Get available URLs by type
        video_urls = []
        audio_urls = []
        image_urls = []
        
        if analyzed_data:
            for filename, analysis in analyzed_data.items():
                if isinstance(analysis, dict):
                    cloud_url = analysis.get("cloud_url", "")
                    file_type = analysis.get("file_type", "")
                    if cloud_url and cloud_url.startswith("https://"):
                        if file_type == "video":
                            video_urls.append(cloud_url)
                        elif file_type == "audio":
                            audio_urls.append(cloud_url)
                        elif file_type == "image":
                            image_urls.append(cloud_url)
        
        # Calculate professional timing based on style
        if pacing == "fast" or style == "viral":
            clip_duration = min(3.0, duration / max(len(video_urls), 1))
        elif pacing == "slow" or style == "cinematic":
            clip_duration = min(5.0, duration / max(len(video_urls), 1))
        else:
            clip_duration = min(4.0, duration / max(len(video_urls), 1))
        
        tracks = []
        
        # VIDEO TRACK - Professional multi-clip edit
        if video_urls:
            video_clips = []
            current_time = 0.0
            
            # Create multiple clips if multiple videos
            for idx, video_url in enumerate(video_urls):
                remaining_time = duration - current_time
                if remaining_time <= 0:
                    break
                
                length = min(clip_duration, remaining_time)
                
                # Vary effects based on style
                effects_map = {
                    "viral": ["zoomInFast", "slideLeftFast", "slideRightFast", None],
                    "cinematic": ["zoomInSlow", "zoomOutSlow", None, None],
                    "motivational": ["zoomIn", "slideUp", None, None],
                    "corporate": [None, "zoomIn", None, None]
                }
                effect_list = effects_map.get(style, [None, "zoomIn", "slideLeft"])
                effect = effect_list[idx % len(effect_list)]
                
                # Vary transitions for visual interest
                transitions_in = ["fade", "slideLeft", "slideRight", "reveal"]
                transitions_out = ["fade", "slideRight", "slideLeft", "fade"]
                
                clip = {
                    "asset": {
                        "type": "video",
                        "src": video_url,
                        "volume": 0.8 if not audio_urls else 0.3  # Duck audio if background music
                    },
                    "start": current_time,
                    "length": length,
                    "fit": "cover",
                    "transition": {
                        "in": transitions_in[idx % len(transitions_in)],
                        "out": transitions_out[idx % len(transitions_out)]
                    }
                }
                
                if effect:
                    clip["effect"] = effect
                
                # Add filter based on style
                if style == "cinematic":
                    clip["filter"] = "contrast"
                elif style == "viral":
                    clip["filter"] = "boost"
                
                video_clips.append(clip)
                current_time += length
            
            tracks.append({"clips": video_clips})
            print(f"✅ Created video track with {len(video_clips)} clip(s)")
        
        # IMAGE TRACK - If no video but images available
        elif image_urls:
            image_clips = []
            current_time = 0.0
            img_duration = duration / len(image_urls) if image_urls else duration
            
            for idx, image_url in enumerate(image_urls):
                remaining_time = duration - current_time
                if remaining_time <= 0:
                    break
                
                length = min(img_duration, remaining_time)
                
                clip = {
                    "asset": {
                        "type": "image",
                        "src": image_url
                    },
                    "start": current_time,
                    "length": length,
                    "fit": "cover",
                    "effect": "zoomInSlow",  # Ken Burns effect
                    "transition": {
                        "in": ["fade", "slideLeft", "slideRight"][idx % 3],
                        "out": "fade"
                    }
                }
                
                image_clips.append(clip)
                current_time += length
            
            tracks.append({"clips": image_clips})
            print(f"✅ Created image track with {len(image_clips)} clip(s)")
        
        # AUDIO TRACK (if available)
        if audio_urls:
            tracks.append({
                "clips": [{
                    "asset": {
                        "type": "audio",
                        "src": audio_urls[0],
                        "volume": 0.6
                    },
                    "start": 0,
                    "length": duration
                }]
            })
            print("✅ Added background audio track")
        
        # TITLE TRACK - Professional multi-title sequence
        titles_data = abstract_plan.get("titles", [])
        if not titles_data:
            # Create default titles based on style
            if style == "viral":
                titles_data = [
                    {"text": "WATCH THIS", "timing": "early"},
                    {"text": "AMAZING", "timing": "climax"}
                ]
            elif style == "cinematic":
                titles_data = [
                    {"text": "A Story", "timing": "early"}
                ]
            else:
                titles_data = [
                    {"text": "Creative Video", "timing": "early"}
                ]
        
        title_clips = []
        for idx, title_info in enumerate(titles_data[:3]):  # Max 3 titles
            text = title_info.get("text", "")
            timing = title_info.get("timing", "mid")
            
            # Calculate timing based on position
            if timing == "early":
                start = 0.5
            elif timing == "climax":
                start = duration * 0.6
            elif timing == "end":
                start = duration - 3.0
            else:
                start = duration * 0.4
            
            # Vary styles for visual interest
            styles = ["blockbuster", "vogue", "future", "subtitle"]
            style_choice = styles[idx % len(styles)]
            
            # Vary sizes
            sizes = ["large", "medium", "x-large"]
            size_choice = sizes[idx % len(sizes)]
            
            title_clips.append({
                "asset": {
                    "type": "title",
                    "text": text,
                    "style": style_choice,
                    "size": size_choice,
                    "color": "#FFFFFF"
                },
                "start": max(0, start),
                "length": 2.5,
                "position": "center",
                "transition": {
                    "in": ["slideUp", "fade", "slideLeft"][idx % 3],
                    "out": "fade"
                }
            })
        
        if title_clips:
            tracks.append({"clips": title_clips})
            print(f"✅ Added title track with {len(title_clips)} title(s)")
        
        # Build final timeline
        timeline = {
            "timeline": {
                "background": "#000000",
                "tracks": tracks
            },
            "output": {
                "format": "mp4",
                "resolution": "hd",
                "fps": 25
            }
        }
        
        print(f"✅ Professional fallback timeline generated: {len(tracks)} tracks, {duration}s duration, {style} style")
        return timeline


# Keep the old class for backward compatibility
class LLMProcessor(EnhancedLLMProcessor):
    """Backward compatibility wrapper"""
    pass
    
    def _build_context(self, analyzed_data: Dict[str, Any], prompt: str) -> str:
        """Build context for LLM based on available data"""
        
        context_parts = [
            f"🎬 CLIENT REQUEST: '{prompt}'",
            "",
            "Create a professional video edit for this request.",
        ]
        
        if analyzed_data:
            context_parts.extend([
                "",
                "📁 AVAILABLE MEDIA:",
            ])
            
            for filename, analysis in analyzed_data.items():
                file_type = analysis.get("file_type", "unknown")
                analysis_text = analysis.get("analysis", "No analysis")
                context_parts.append(f"• {filename} ({file_type.upper()}): {analysis_text}")
            
            context_parts.extend([
                "",
                "🎯 INSTRUCTIONS:",
                "• Use the analyzed media content intelligently",
                "• Create timeline that matches the media characteristics",
                "• Apply professional editing techniques",
            ])
        else:
            context_parts.extend([
                "",
                "📝 NOTE: No media files provided - create conceptual timeline",
            ])
        
        context_parts.extend([
            "",
            "📋 RESPONSE FORMAT:",
            "Return JSON with:",
            '• "content": Brief explanation of your creative approach',
            '• "json_plan": Complete Shotstack timeline JSON',
        ])
        
        return "\n".join(context_parts)


class AdvancedJSONExtractor:
    """Advanced JSON extraction system with multiple fallback strategies"""
    
    def __init__(self):
        print("📄 AdvancedJSONExtractor initialized with intelligent parsing")
        self.extraction_attempts = 0
        self.successful_extractions = 0
        
        # Import regex for advanced parsing
        import re
        self.re = re
    
    def extract_and_separate(self, llm_response: str) -> tuple[str, Dict[str, Any]]:
        """Extract content and JSON plan with multiple fallback strategies"""
        
        self.extraction_attempts += 1
        print(f"🔍 JSON Extraction Attempt #{self.extraction_attempts}")
        
        # DEBUG: Show the raw LLM response (first 500 chars)
        if llm_response:
            preview = llm_response[:500].replace('\n', '\\n')
            print(f"📋 LLM Response Preview: {preview}...")
        else:
            print("⚠️ LLM response is empty!")
        
        # Strategy 1: Direct JSON parsing
        result = self._try_direct_json_parsing(llm_response)
        if result:
            self.successful_extractions += 1
            return result
        
        # Strategy 1.5: Extract from ```json ... ``` code block
        result = self._try_codeblock_json(llm_response)
        if result:
            self.successful_extractions += 1
            return result
        
        # Strategy 2: Clean and parse
        result = self._try_cleaned_json_parsing(llm_response)
        if result:
            self.successful_extractions += 1
            return result
        
        # Strategy 3: Regex extraction
        result = self._try_regex_extraction(llm_response)
        if result:
            self.successful_extractions += 1
            return result
        
        # Strategy 4: Intelligent content detection
        result = self._try_intelligent_content_detection(llm_response)
        if result:
            self.successful_extractions += 1
            return result
        
        # Strategy 5: Response repair
        result = self._try_response_repair(llm_response)
        if result:
            self.successful_extractions += 1
            return result
        
        # Strategy 6: Generate fallback JSON
        return self._generate_fallback_response(llm_response)
    
    def _try_direct_json_parsing(self, response: str) -> tuple[str, Dict[str, Any]] | None:
        """Strategy 1: Try direct JSON parsing"""
        try:
            if not response or not response.strip():
                print("⚠️ Empty response received")
                return None
            
            response_json = json.loads(response.strip())
            content = response_json.get("content", "AI generated content")
            json_plan = response_json.get("json_plan", {})
            
            print("✅ Direct JSON parsing successful")
            return content, json_plan
            
        except json.JSONDecodeError:
            print("🔄 Direct parsing failed, trying next strategy...")
            return None
        except Exception as e:
            print(f"🔄 Direct parsing error: {str(e)}")
            return None
    
    def _try_cleaned_json_parsing(self, response: str) -> tuple[str, Dict[str, Any]] | None:
        """Strategy 2: Clean response and parse"""
        try:
            cleaned = self._advanced_clean_response(response)
            if not cleaned:
                return None
            
            response_json = json.loads(cleaned)
            content = response_json.get("content", "AI generated content")
            json_plan = response_json.get("json_plan", {})
            
            print("✅ Cleaned JSON parsing successful")
            return content, json_plan
            
        except json.JSONDecodeError:
            print("🔄 Cleaned parsing failed, trying next strategy...")
            return None
        except Exception as e:
            print(f"🔄 Cleaned parsing error: {str(e)}")
            return None
    
    def _try_codeblock_json(self, response: str) -> tuple[str, Dict[str, Any]] | None:
        """Extract JSON enclosed in ```json ... ``` or ``` ... ``` blocks"""
        try:
            if not response:
                return None
            # Look for fenced code blocks with optional json hint
            pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
            m = self.re.search(pattern, response, self.re.IGNORECASE)
            if m:
                block = m.group(1)
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    content = parsed.get("content", "AI generated content")
                    json_plan = parsed.get("json_plan", {})
                    print("✅ Code-block JSON extraction successful")
                    return content, json_plan
            return None
        except Exception as e:
            print(f"🔄 Code-block extraction error: {str(e)}")
            return None

    def _try_regex_extraction(self, response: str) -> tuple[str, Dict[str, Any]] | None:
        """Strategy 3: Extract a JSON object from mixed content using brace balancing"""
        try:
            candidate = self._extract_largest_json_object(response)
            if candidate:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    content = parsed.get("content", "AI generated content")
                    json_plan = parsed.get("json_plan", {})
                    print("✅ Brace-balanced extraction successful")
                    return content, json_plan
            print("🔄 Regex extraction failed, trying next strategy...")
            return None
        
        except Exception as e:
            error_msg = str(e)
            print(f"🔄 Regex extraction error: {error_msg[:100]}")
            # Log the problematic area if JSON parsing failed
            if "line" in error_msg.lower() and "column" in error_msg.lower():
                print(f"💡 Hint: JSON syntax error detected - likely invalid characters or malformed structure")
            return None

    def _extract_largest_json_object(self, text: str) -> str | None:
        """Find the first plausible JSON object by tracking braces, allowing nested objects."""
        if not text:
            return None
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None
    
    def _try_intelligent_content_detection(self, response: str) -> tuple[str, Dict[str, Any]] | None:
        """Strategy 4: Intelligently detect content and JSON separately"""
        try:
            # Look for content and json_plan patterns
            content_pattern = r'"content"\s*:\s*"([^"]*)"'
            json_plan_pattern = r'"json_plan"\s*:\s*(\{.*?\})'
            
            content_match = self.re.search(content_pattern, response, self.re.DOTALL)
            json_plan_match = self.re.search(json_plan_pattern, response, self.re.DOTALL)
            
            if content_match or json_plan_match:
                content = content_match.group(1) if content_match else "AI generated content"
                
                json_plan = {}
                if json_plan_match:
                    try:
                        json_plan = json.loads(json_plan_match.group(1))
                    except json.JSONDecodeError:
                        json_plan = {}
                
                print("✅ Intelligent content detection successful")
                return content, json_plan
            
            print("🔄 Content detection failed, trying next strategy...")
            return None
            
        except Exception as e:
            print(f"🔄 Content detection error: {str(e)}")
            return None
    
    def _try_response_repair(self, response: str) -> tuple[str, Dict[str, Any]] | None:
        """Strategy 5: Attempt to repair malformed JSON"""
        try:
            # Common JSON repair strategies
            repaired = response.strip()
            
            # Fix common issues
            repaired = self.re.sub(r',\s*}', '}', repaired)  # Remove trailing commas
            repaired = self.re.sub(r',\s*]', ']', repaired)  # Remove trailing commas in arrays
            repaired = self.re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', repaired)  # Quote unquoted keys
            
            # Try to balance braces
            open_braces = repaired.count('{')
            close_braces = repaired.count('}')
            if open_braces > close_braces:
                repaired += '}' * (open_braces - close_braces)
            
            response_json = json.loads(repaired)
            content = response_json.get("content", "AI generated content")
            json_plan = response_json.get("json_plan", {})
            
            print("✅ Response repair successful")
            return content, json_plan
            
        except Exception as e:
            print(f"🔄 Response repair failed: {str(e)}")
            return None
    
    def _generate_fallback_response(self, response: str) -> tuple[str, Dict[str, Any]]:
        """Strategy 6: Generate intelligent fallback when all else fails"""
        print("🛠️ Generating intelligent fallback response...")
        
        # Analyze the response to extract any useful information
        content_lines = []
        
        if response and response.strip():
            # Try to extract meaningful content from the response
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('}'):
                    content_lines.append(line)
        
        if content_lines:
            fallback_content = f"""🔧 **Partial Response Recovered**

The AI provided the following response, but it wasn't in the expected JSON format:

{chr(10).join(content_lines[:5])}  # Show first 5 lines

I've generated a basic video editing plan based on the available information."""
        else:
            fallback_content = """🔧 **Fallback Response Generated**

The AI response couldn't be parsed, but I've created a basic video editing plan for you. You can refine this plan by providing more specific instructions."""
        
        # Generate a basic JSON plan
        fallback_json_plan = {
            "timeline": {
                "background": "#000000",
                "tracks": [
                    {
                        "type": "video",
                        "clips": [
                            {
                                "asset": {
                                    "type": "video",
                                    "src": "input_video.mp4",
                                    "volume": 1.0
                                },
                                "start": 0,
                                "length": 30,
                                "transition": {
                                    "in": {"type": "fade", "duration": 1},
                                    "out": {"type": "fade", "duration": 1}
                                }
                            }
                        ]
                    }
                ]
            },
            "output": {
                "format": "mp4",
                "resolution": "1920x1080",
                "fps": 30
            }
        }
        
        print("✅ Fallback response generated successfully")
        return fallback_content, fallback_json_plan
    
    def _advanced_clean_response(self, response: str) -> str:
        """Advanced response cleaning with multiple strategies"""
        if not response:
            return ""
        
        cleaned = response.strip()
        
        # Remove markdown code fences anywhere in the text
        cleaned = self.re.sub(r"```json", "", cleaned, flags=self.re.IGNORECASE)
        cleaned = self.re.sub(r"```", "", cleaned)
        cleaned = self.re.sub(r"^`|`$", "", cleaned)

        # Remove common prefixes that might interfere
        prefixes_to_remove = [
            'Here is the JSON:',
            'Here\'s the JSON:',
            'JSON:',
            'Response:',
            'Here is the response:',
            'Here\'s the response:'
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove HTML tags if present
        cleaned = self.re.sub(r'<[^>]+>', '', cleaned)

        # If there is a JSON object inside, trim to it
        first = cleaned.find('{')
        last = cleaned.rfind('}')
        if first != -1 and last != -1 and last > first:
            candidate = cleaned[first:last+1]
            
            # Remove JavaScript-style comments (// ...) that break JSON parsing
            # This is crucial as LLM often adds comments despite instructions
            lines = candidate.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove inline comments but preserve strings with //
                in_string = False
                cleaned_line = []
                i = 0
                while i < len(line):
                    char = line[i]
                    if char == '"' and (i == 0 or line[i-1] != '\\'):
                        in_string = not in_string
                        cleaned_line.append(char)
                    elif char == '/' and i + 1 < len(line) and line[i+1] == '/' and not in_string:
                        # Found comment, stop processing this line
                        break
                    else:
                        cleaned_line.append(char)
                    i += 1
                cleaned_lines.append(''.join(cleaned_line))
            
            candidate = '\n'.join(cleaned_lines)
            
            # Remove trailing commas before } or ] (invalid JSON)
            candidate = self.re.sub(r',(\s*[}\]])', r'\1', candidate)
            
            # Fix array syntax errors: ],[{ should be },{ (common LLM mistake)
            # Example: "tracks":[{...}],[{...}] → "tracks":[{...},{...}]
            candidate = self.re.sub(r'\]\s*,\s*\[', ',', candidate)
            
            return candidate.strip()
        
        return cleaned.strip()
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        success_rate = (self.successful_extractions / self.extraction_attempts * 100) if self.extraction_attempts > 0 else 0
        
        return {
            "total_attempts": self.extraction_attempts,
            "successful_extractions": self.successful_extractions,
            "success_rate": f"{success_rate:.1f}%"
        }


# Keep the old class for backward compatibility
class JSONExtractor(AdvancedJSONExtractor):
    """Backward compatibility wrapper"""
    pass


class ShotstackRenderer:
    """Handle Shotstack API rendering"""
    
    # Valid Shotstack enums
    VALID_TITLE_STYLES = {
        "minimal", "blockbuster", "vogue", "sketchy", "skinny", 
        "chunk", "chunkLight", "marker", "future", "subtitle"
    }
    VALID_FILTERS = {
        "none", "blur", "boost", "contrast", "darken", 
        "greyscale", "lighten", "muted", "negative"
    }
    
    def __init__(self):
        self.headers = {
            "x-api-key": SHOTSTACK_API_KEY,
            "Content-Type": "application/json"
        }
        self.configured = SHOTSTACK_API_KEY != "YOUR_SHOTSTACK_API_KEY"
        print(f"🎬 ShotstackRenderer initialized ({'Configured' if self.configured else 'Demo mode'})")
    
    def _fix_invalid_values(self, json_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common invalid values before sending to Shotstack"""
        fixed_count = 0
        
        if "timeline" in json_plan and "tracks" in json_plan["timeline"]:
            for track in json_plan["timeline"]["tracks"]:
                if "clips" in track:
                    for clip in track["clips"]:
                        if "asset" in clip and isinstance(clip["asset"], dict):
                            asset = clip["asset"]
                            
                            # Fix invalid title styles
                            if asset.get("type") == "title" and "style" in asset:
                                if asset["style"] not in self.VALID_TITLE_STYLES:
                                    old_style = asset["style"]
                                    asset["style"] = "minimal"  # Default fallback
                                    print(f"⚠️ Fixed invalid title style: '{old_style}' → 'minimal'")
                                    fixed_count += 1
                            
                            # Fix invalid filters
                            if "filter" in clip:
                                if clip["filter"] not in self.VALID_FILTERS:
                                    old_filter = clip["filter"]
                                    clip["filter"] = "none"
                                    print(f"⚠️ Fixed invalid filter: '{old_filter}' → 'none'")
                                    fixed_count += 1
        
        if fixed_count > 0:
            print(f"✅ Auto-fixed {fixed_count} invalid value(s)")
        
        return json_plan
    
    def render(self, json_plan: Dict[str, Any]) -> Optional[str]:
        """Send JSON to Shotstack for rendering"""
        
        if not self.configured:
            print("⚠️ Shotstack not configured - demo mode")
            return None
        
        if not json_plan:
            print("❌ No JSON plan to render")
            return None
        
        try:
            # Normalize and fix invalid values
            json_plan = self._normalize_plan(json_plan)
            json_plan = self._fix_invalid_values(json_plan)
            
            # DEBUG: Print the exact JSON being sent to Shotstack
            import json as json_module
            print("=" * 80)
            print("📤 SENDING TO SHOTSTACK:")
            print(json_module.dumps(json_plan, indent=2))
            print("=" * 80)
            
            # Send render request
            response = requests.post(
                f"{SHOTSTACK_BASE_URL}/render",
                headers=self.headers,
                json=json_plan,
                timeout=60
            )
            
            if response.status_code == 201:
                render_data = response.json()
                render_id = render_data["response"]["id"]
                print(f"🎬 Render started: {render_id}")
                
                # Poll for completion (simplified)
                video_url = self._poll_render_status(render_id)
                return video_url
            else:
                print(f"❌ Render failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print("=" * 80)
                    print("📋 FULL ERROR RESPONSE:")
                    import json as json_module
                    print(json_module.dumps(error_data, indent=2))
                    print("=" * 80)
                    
                    # Extract specific validation errors
                    if "response" in error_data and "errors" in error_data["response"]:
                        print("\n🚨 VALIDATION ERRORS:")
                        for err in error_data["response"]["errors"]:
                            print(f"  • Field: {err.get('field')}")
                            print(f"    Error: {err.get('message')}")
                            print(f"    Invalid value: {err.get('context', {}).get('value')}")
                            print()
                except Exception as e:
                    print(f"Error response body: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Render error: {str(e)}")
            return None
    
    def _poll_render_status(self, render_id: str) -> Optional[str]:
        """Poll render status until completion"""
        max_attempts = 60  # 5 minutes max
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{SHOTSTACK_BASE_URL}/render/{render_id}",
                    headers=self.headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data["response"]["status"]
                    
                    if status == "done":
                        video_url = data["response"]["url"]
                        print(f"✅ Render completed: {video_url}")
                        return video_url
                    elif status == "failed":
                        print("❌ Render failed")
                        print("=" * 80)
                        print("📋 FULL FAILURE RESPONSE:")
                        import json as json_module
                        print(json_module.dumps(data, indent=2))
                        print("=" * 80)
                        
                        # Extract specific error message if available
                        if "response" in data:
                            error_msg = data["response"].get("error")
                            if error_msg:
                                print(f"\n🚨 ERROR MESSAGE: {error_msg}")
                            
                            # Check for validation errors
                            if "errors" in data["response"]:
                                print("\n🚨 VALIDATION ERRORS:")
                                for err in data["response"]["errors"]:
                                    print(f"  • Field: {err.get('field')}")
                                    print(f"    Error: {err.get('message')}")
                                    if 'context' in err:
                                        print(f"    Context: {err.get('context')}")
                                    print()
                        return None
                    else:
                        print(f"⏳ Rendering... ({status})")
                        time.sleep(5)
                
            except Exception as e:
                print(f"❌ Polling error: {str(e)}")
                import traceback
                print("📋 Full traceback:")
                traceback.print_exc()
                return None
        
        print("⏰ Render timeout")
        return None

    # Compatibility methods to match orchestrator expectations
    def render_video(self, json_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper that aligns with main_video_agent expectations.
        Returns a dict with status, render_id, video_url. This implementation uses
        the synchronous render() above; when configured=False, acts in demo mode.
        """
        if not self.configured:
            return {
                "status": "demo_mode",
                "message": "Demo mode - Shotstack API key not configured",
                "render_id": f"demo_{int(time.time())}",
                "video_url": None
            }

        url = self.render(json_plan)
        if url:
            return {
                "status": "completed",
                "message": "Render completed successfully",
                "render_id": None,
                "video_url": url
            }
        else:
            return {
                "status": "error",
                "message": "Render failed - Check console logs above for detailed error information",
                "render_id": None,
                "video_url": None
            }

    # ---------------------------
    # JSON Normalization Helpers
    # ---------------------------
    def _verify_no_nested_structures(self, plan: Dict[str, Any]) -> None:
        """Final verification that no nested timeline/tracks/clips exist in assets"""
        def check_asset(asset: Any, path: str = "asset") -> None:
            if isinstance(asset, dict):
                forbidden = ["timeline", "tracks", "clips", "output", "content", "json_plan"]
                for key in forbidden:
                    if key in asset:
                        print(f"🚨 CRITICAL: Found nested '{key}' in {path} - THIS SHOULD NOT HAPPEN!")
                        print(f"   Asset content: {str(asset)[:200]}")
                
                # Check nested values
                for key, value in asset.items():
                    check_asset(value, f"{path}.{key}")
            elif isinstance(asset, list):
                for i, item in enumerate(asset):
                    check_asset(item, f"{path}[{i}]")
        
        # Check all assets in all clips
        for track_idx, track in enumerate(plan.get("timeline", {}).get("tracks", [])):
            for clip_idx, clip in enumerate(track.get("clips", [])):
                asset = clip.get("asset")
                if asset:
                    check_asset(asset, f"timeline.tracks[{track_idx}].clips[{clip_idx}].asset")
    
    def _clean_asset_recursively(self, obj: Any, depth: int = 0) -> Any:
        """Recursively clean any object to remove nested timeline/tracks/clips structures.
        This is critical because LLMs sometimes generate deeply nested invalid structures.
        """
        import copy
        
        if isinstance(obj, dict):
            # Create a new dict to ensure we're not modifying references
            cleaned = {}
            # Forbidden keys that should NEVER be in assets
            forbidden = [
                "timeline", "tracks", "clips", "output", "content", "json_plan",
                "volumeEffect", "speed", "crop", "transcode"
            ]
            
            # chromaKey is ONLY allowed in video assets
            if obj.get("type") not in ["video", None] and "chromaKey" in obj:
                errors.append(f"chromaKey is only valid in video assets, found in {obj.get('type', 'unknown')} asset")
            
            # Validate chromaKey structure if present
            if "chromaKey" in obj and isinstance(obj["chromaKey"], dict):
                ck = obj["chromaKey"]
                if "threshold" in ck and not isinstance(ck["threshold"], (int, float)):
                    errors.append("chromaKey.threshold must be a number (0-255)")
                if "halo" in ck and not isinstance(ck["halo"], (int, float)):
                    errors.append("chromaKey.halo must be a number (0-100)")
            
            for key, value in obj.items():
                # Skip forbidden keys entirely
                if key in forbidden:
                    if depth == 0:  # Only print at top level to avoid spam
                        print(f"⚠️ RECURSIVE CLEANUP: Removing unsupported '{key}' from asset")
                    continue
                
                # Recursively clean the value
                cleaned[key] = self._clean_asset_recursively(value, depth + 1)
            
            return cleaned
        elif isinstance(obj, list):
            return [self._clean_asset_recursively(item, depth + 1) for item in obj]
        else:
            return obj
    
    def _normalize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through the LLM's plan directly without validation"""
        print("🔧 Passing timeline directly to Shotstack (no validation)")
        import copy
        fixed = copy.deepcopy(plan) if isinstance(plan, dict) else {}

        # Ensure structure
        tl = fixed.setdefault("timeline", {})
        tracks = tl.get("tracks")
        if isinstance(tracks, dict):
            tl["tracks"] = [tracks]
        elif not isinstance(tracks, list):
            tl["tracks"] = []

        for t in tl.get("tracks", []):
            if not isinstance(t, dict):
                continue
            # Remove track-level type field (not used by Shotstack)
            t.pop("type", None)

            clips = t.get("clips")
            if isinstance(clips, dict):
                t["clips"] = [clips]
                clips = t["clips"]
            elif not isinstance(clips, list):
                t["clips"] = []
                clips = t["clips"]

            # NO VALIDATION - pass clips through as-is
            pass

        # Return plan as-is without any modifications
        return fixed

    def check_render_status(self, render_id: str) -> Dict[str, Any]:
        """Provide a status response. In demo mode, return a fake completed job.
        In configured mode, this simplified renderer uses synchronous polling, so no render_id.
        """
        if not self.configured:
            return {
                "status": "demo_mode",
                "progress": 100,
                "video_url": "https://example.com/demo_video.mp4",
                "message": "Demo mode - no actual rendering"
            }
        return {
            "status": "not_supported",
            "progress": 0,
            "video_url": None,
            "message": "Synchronous render flow used; no render_id available"
        }


print("✅ Clean modular components loaded successfully!")
