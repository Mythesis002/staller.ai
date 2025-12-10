#!/usr/bin/env python
"""
Minimal FastAPI server exposing a synchronous /plan endpoint that reuses the
existing VideoEditingAgent from main_video_agent.py. This is intended for quick
MVP testing (no DB, no workers).

Run:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Test:
  curl -X POST http://127.0.0.1:8000/plan \
    -H "Content-Type: application/json" \
    -d '{"prompt": "make a cinematic 15s teaser", "media_files": []}'
"""

from typing import List, Optional, Any, Dict
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
import requests
import tempfile
import shutil

# Import existing orchestrator
from main_video_agent import VideoEditingAgent

app = FastAPI(title="Video Editing Agent API (MVP)")
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
agent = VideoEditingAgent()

# Simple in-memory cache to avoid re-analyzing the same file repeatedly
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
# Simple in-memory session state to support iterative refinement
SESSIONS: Dict[str, Dict[str, Any]] = {}
# In-flight guard to prevent concurrent duplicate analysis of the same path
from threading import Lock
_ANALYSIS_LOCK = Lock()
ANALYSIS_INFLIGHT = set()

class CachedAnalyser:
    def __init__(self, inner, cache: Dict[str, Dict[str, Any]]):
        self.inner = inner
        self.cache = cache

    def _key(self, path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _get_cached(self, path: str):
        k = self._key(path)
        if k in self.cache:
            res = dict(self.cache[k])
            res.setdefault("status", res.get("status", "ok"))
            res["cached"] = True
            return res
        return None

    def _store(self, path: str, res: Dict[str, Any]):
        try:
            self.cache[self._key(path)] = res
        except Exception:
            pass

    def analyze_video(self, path: str):
        cached = self._get_cached(path)
        if cached is not None:
            return cached
        res = self.inner.analyze_video(path)
        if isinstance(res, dict):
            self._store(path, res)
        return res

    def analyze_audio(self, path: str):
        cached = self._get_cached(path)
        if cached is not None:
            return cached
        res = self.inner.analyze_audio(path)
        if isinstance(res, dict):
            self._store(path, res)
        return res

    def analyze_image(self, path: str):
        cached = self._get_cached(path)
        if cached is not None:
            return cached
        res = self.inner.analyze_image(path)
        if isinstance(res, dict):
            self._store(path, res)
        return res

    # Some agent code may call a generic analyze(path) method.
    def analyze(self, path: str):
        cached = self._get_cached(path)
        if cached is not None:
            return cached
        # If the inner has a generic analyze, prefer that
        try:
            if hasattr(self.inner, 'analyze') and callable(getattr(self.inner, 'analyze')):
                res = self.inner.analyze(path)
                if isinstance(res, dict):
                    self._store(path, res)
                return res
        except Exception:
            pass
        # Fallback to routing by extension
        ext = Path(path).suffix.lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
            return self.analyze_video(path)
        if ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']:
            return self.analyze_audio(path)
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return self.analyze_image(path)
        # Unknown -> call inner.analyze if present or return a skipped dict
        if hasattr(self.inner, 'analyze') and callable(getattr(self.inner, 'analyze')):
            res = self.inner.analyze(path)
            if isinstance(res, dict):
                self._store(path, res)
            return res
        return {"status": "skipped", "message": f"Unsupported type {ext or 'unknown'}", "cached": False}


class CachedOnlyAnalyser:
    """Analyser that serves only from cache and never calls the inner analyser.
    Used when the client indicates media was already analyzed (skip_media=1).
    """
    def __init__(self, inner, cache: Dict[str, Dict[str, Any]]):
        self.inner = inner
        self.cache = cache

    def _key(self, path: str) -> str:
        try:    
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _from_cache(self, path: str):
        k = self._key(path)
        if k in self.cache:
            res = dict(self.cache[k])
            res.setdefault("status", res.get("status", "ok"))
            res["cached"] = True
            return res
        # Return a benign minimal analysis to ensure no downstream analyzer is invoked
        return {"status": "ok", "message": "analysis skipped (skip_media)", "cached": False}

    def analyze(self, path: str):
        return self._from_cache(path)

    def analyze_video(self, path: str):
        return self._from_cache(path)

    def analyze_audio(self, path: str):
        return self._from_cache(path)

    def analyze_image(self, path: str):
        return self._from_cache(path)

# Wrap the agent's analyser so downstream calls reuse cached results
try:
    agent.media_analyser = CachedAnalyser(agent.media_analyser, ANALYSIS_CACHE)
except Exception:
    # If the agent does not expose media_analyser as expected, continue without caching
    pass


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the minimal web UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
def favicon():
    # Simple placeholder to avoid noisy 404s
    return HTMLResponse(content="", status_code=204)


class PlanRequest(BaseModel):
    prompt: str = Field(..., description="User instruction for the video plan")
    media_files: Optional[List[str]] = Field(default_factory=list, description="Absolute paths to local media files")
    media_urls: Optional[List[str]] = Field(default_factory=list, description="Cloud URLs corresponding to media files (same order)")


@app.post("/plan")
def create_plan(req: PlanRequest) -> Dict[str, Any]:
    """Create a new editing plan by invoking the VideoEditingAgent."""
    try:
        # Build url_mappings if media_urls provided
        url_mappings = {}
        if req.media_files and req.media_urls and len(req.media_files) == len(req.media_urls):
            for file_path, cloud_url in zip(req.media_files, req.media_urls):
                if cloud_url:
                    filename = Path(file_path).name
                    url_mappings[filename] = cloud_url
                    print(f"🔗 API: Mapping {filename} → {cloud_url[:60]}...")
        
        result = agent.process_request(req.prompt, req.media_files or [], url_mappings=url_mappings if url_mappings else None)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Unexpected agent response")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Client-side direct upload signing endpoint
@app.get("/media/cloudinary/sign")
def cloudinary_sign() -> Dict[str, Any]:
    """Provide a short-lived signature for direct client uploads to Cloudinary.
    Returns api_key, cloud_name, timestamp, folder, and signature.
    """
    try:
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")
        folder = os.getenv("CLOUDINARY_FOLDER", "videoagent/uploads")
        if not (cloud_name and api_key and api_secret):
            raise HTTPException(status_code=400, detail="Missing Cloudinary credentials in env")
        import time as _t, hashlib
        timestamp = int(_t.time())
        params = { 'timestamp': str(timestamp), 'folder': folder }
        to_sign = '&'.join([f"{k}={v}" for k, v in sorted(params.items())]) + api_secret
        signature = hashlib.sha1(to_sign.encode('utf-8')).hexdigest()
        return {
            'cloud_name': cloud_name,
            'api_key': api_key,
            'timestamp': timestamp,
            'folder': folder,
            'signature': signature,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AnalyzeRequest(BaseModel):
    media_files: List[str] = Field(..., description="Absolute paths to local media files")


@app.post("/media/analyze")
def analyze_media(req: AnalyzeRequest) -> Dict[str, Any]:
    """Analyze provided media files using the agent's media analyzer.
    Returns a mapping filename -> analysis dict similar to the Tkinter GUI."""
    try:
        analyses: Dict[str, Any] = {}
        for fp in req.media_files:
            if not fp:
                continue
            p = Path(fp)
            if not p.exists():
                analyses[p.name] = {"status": "error", "message": "File not found"}
                continue
            ext = p.suffix.lower()
            try:
                key = str(p.resolve())
                # Deduplicate concurrent analyses
                with _ANALYSIS_LOCK:
                    if key in ANALYSIS_INFLIGHT:
                        analyses[p.name] = {"status": "in_progress", "message": "Analysis already running"}
                        continue
                    ANALYSIS_INFLIGHT.add(key)
                try:
                    if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
                        res = agent.media_analyser.analyze_video(str(p))
                    elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']:
                        res = agent.media_analyser.analyze_audio(str(p))
                    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                        res = agent.media_analyser.analyze_image(str(p))
                    else:
                        res = {"status": "skipped", "message": f"Unsupported type {ext}"}
                finally:
                    with _ANALYSIS_LOCK:
                        ANALYSIS_INFLIGHT.discard(key)
                # Normalize to dict and store in cache
                res_dict = res if isinstance(res, dict) else getattr(res, '__dict__', {"status":"ok","analysis":str(res)})
                analyses[p.name] = res_dict
                ANALYSIS_CACHE[str(p.resolve())] = res_dict
            except Exception as e:
                analyses[p.name] = {"status": "error", "message": str(e)}
        return {"status": "success", "analyses": analyses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeRemoteRequest(BaseModel):
    url: str = Field(..., description="HTTPS URL to media (Cloudinary)")
    filename: Optional[str] = Field(default=None, description="Optional filename hint")


def _validate_cloud_url(u: str) -> None:
    from urllib.parse import urlparse
    pu = urlparse(u)
    if pu.scheme not in {"https"}:
        raise HTTPException(status_code=400, detail="Only HTTPS URLs are allowed")
    host = pu.netloc.lower()
    path = pu.path or ""
    allowed_cloud = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    allowed_hosts_env = os.getenv("ALLOWED_MEDIA_HOSTS", "")
    allowed_hosts = [h.strip().lower() for h in allowed_hosts_env.split(",") if h.strip()]
    # Accept explicit allowlist
    if allowed_hosts and any(host.endswith(h) or host == h for h in allowed_hosts):
        return
    # Accept Cloudinary default domain with correct cloud name in path: /<cloud_name>/...
    if host.endswith("res.cloudinary.com"):
        if not allowed_cloud:
            return
        # path like /<cloud_name>/...
        if path.startswith(f"/{allowed_cloud}/"):
            return
        raise HTTPException(status_code=400, detail="Cloudinary URL cloud name mismatch")
    # Fallback: if CLOUDINARY_CLOUD_NAME is set and present in host (custom domains)
    if allowed_cloud and allowed_cloud in host:
        return
    raise HTTPException(status_code=400, detail="URL host not allowed")


@app.post("/media/analyze_remote")
def analyze_remote(req: AnalyzeRemoteRequest) -> Dict[str, Any]:
    """Analyze a remote media URL by downloading to a temp file and invoking the analyzer.
    Caches results by URL to avoid re-analysis.
    """
    try:
        _validate_cloud_url(req.url)
        # Dedup inflight by URL key
        key = req.url
        with _ANALYSIS_LOCK:
            if key in ANALYSIS_INFLIGHT:
                return {"status": "in_progress", "message": "Analysis already running"}
            ANALYSIS_INFLIGHT.add(key)
        try:
            # Serve from cache if available
            if key in ANALYSIS_CACHE:
                res = dict(ANALYSIS_CACHE[key])
                res.setdefault("status", res.get("status", "ok"))
                res["cached"] = True
                return {"status": "success", "analysis": res}

            # Download to temp
            r = requests.get(req.url, stream=True, timeout=240)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Download failed: {r.status_code} {r.text[:200] if hasattr(r, 'text') else ''}")
            suffix = Path(req.filename or Path(req.url).name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tmp_path = Path(tf.name)
                shutil.copyfileobj(r.raw, tf)

            # Route to proper analyzer based on extension
            ext = tmp_path.suffix.lower()
            if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
                res = agent.media_analyser.analyze_video(str(tmp_path))
            elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']:
                res = agent.media_analyser.analyze_audio(str(tmp_path))
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                res = agent.media_analyser.analyze_image(str(tmp_path))
            else:
                res = {"status": "skipped", "message": f"Unsupported type {ext}"}

            # Normalize and cache by URL
            res_dict = res if isinstance(res, dict) else getattr(res, '__dict__', {"status":"ok","analysis":str(res)})
            ANALYSIS_CACHE[key] = res_dict
            
            # PERSIST TO DISK to avoid re-analysis later!
            try:
                from video_editing_agent import MediaAnalysis
                import datetime
                
                filename = req.filename or Path(req.url).name
                
                media_analysis = MediaAnalysis(
                    file_path=req.url,  # Use cloud URL as file_path (primary identifier)
                    file_type=res_dict.get("file_type", "unknown"),
                    filename=filename,
                    analysis=res_dict.get("analysis", ""),
                    metadata=res_dict.get("metadata", {}),
                    timestamp=datetime.datetime.now().isoformat(),
                    status=res_dict.get("status", "success"),
                    cloud_url=req.url  # Store cloud URL
                )
                
                agent.memory.store_analysis(media_analysis)
                print(f"💾 Stored analysis for {filename} → {req.url[:60]}...")
            except Exception as persist_err:
                print(f"⚠️ Failed to persist analysis: {persist_err}")
            
            return {"status": "success", "analysis": res_dict}
        finally:
            with _ANALYSIS_LOCK:
                ANALYSIS_INFLIGHT.discard(key)
            # Cleanup temp file if created
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CloudinaryUploadRequest(BaseModel):
    path: str = Field(..., description="Absolute path to local file to upload")


@app.post("/media/cloudinary")
def upload_cloudinary(req: CloudinaryUploadRequest) -> Dict[str, Any]:
    """Upload a local file to Cloudinary using unsigned uploads.
    Requires env: CLOUDINARY_CLOUD_NAME and CLOUDINARY_UPLOAD_PRESET.
    Returns cloudinary_url and public_id for future rendering."""
    try:
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")
        upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET")
        folder = os.getenv("CLOUDINARY_FOLDER", "videoagent/uploads")
        if not cloud_name:
            raise HTTPException(status_code=400, detail="Missing CLOUDINARY_CLOUD_NAME env var")
        p = Path(req.path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="File not found")
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/auto/upload"
        with p.open('rb') as f:
            files = { 'file': (p.name, f) }

            # If API secret is present, perform signed upload; else fallback to unsigned preset
            if api_key and api_secret:
                import hashlib, time as _t
                timestamp = int(_t.time())
                # Params to sign (exclude file, api_key, signature)
                params = {
                    'timestamp': str(timestamp),
                    'folder': folder,
                }
                # Build signature string: key=value sorted by key joined with & then + api_secret
                to_sign = '&'.join([f"{k}={v}" for k, v in sorted(params.items())]) + api_secret
                signature = hashlib.sha1(to_sign.encode('utf-8')).hexdigest()

                data = {
                    'api_key': api_key,
                    'timestamp': timestamp,
                    'signature': signature,
                    'folder': folder,
                }
            else:
                if not upload_preset:
                    raise HTTPException(status_code=400, detail="Missing CLOUDINARY_UPLOAD_PRESET for unsigned uploads")
                data = { 'upload_preset': upload_preset, 'folder': folder }

            r = requests.post(url, files=files, data=data, timeout=120)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            j = r.json()
            return {
                "status": "success",
                "cloudinary_url": j.get("secure_url") or j.get("url"),
                "public_id": j.get("public_id"),
                "resource_type": j.get("resource_type"),
                "bytes": j.get("bytes"),
                "version": j.get("version"),
                "signature_used": bool(api_key and api_secret),
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# Server-Sent Events (SSE)
# ==========================

def _sse_format(data: Dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    import json as _json
    return f"data: {_json.dumps(data)}\n\n"


@app.get("/stream/plan")
def stream_plan(prompt: str, media: Optional[str] = None, skip_media: int = 0, media_urls: Optional[str] = None):
    """SSE stream for planning + optional rendering.
    Query params:
      - prompt: text
      - media: optional, '|' separated absolute paths
    """
    media_files = [m for m in (media or "").split("|") if m]
    media_urls_list = [u for u in (media_urls or "").split("|") if u]

    def event_gen():
        try:
            original_analyser = agent.media_analyser
            if skip_media and not media_urls_list:
                agent.media_analyser = CachedOnlyAnalyser(original_analyser, ANALYSIS_CACHE)
            
            # If media URLs are provided, download to temp files for processing
            temp_files = []
            filename_to_url = {}
            try:
                local_inputs = list(media_files)
                
                # STEP 1: Media Analysis
                if (media_files or media_urls_list) and not skip_media:
                    yield _sse_format({"type": "step", "message": "Analyzing media…"})
                
                if media_urls_list:
                    for u in media_urls_list:
                        _validate_cloud_url(u)
                        r = requests.get(u, stream=True, timeout=240)
                        if r.status_code != 200:
                            continue
                        suffix = Path(u).suffix or ".bin"
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        with tf as f:
                            shutil.copyfileobj(r.raw, f)
                        p = Path(tf.name)
                        temp_files.append(p)
                        local_inputs.append(str(p))
                        filename_to_url[p.name] = u

                # STEP 2: Prompt Enhancement
                yield _sse_format({"type": "step", "message": "Enhancing prompt…"})
                
                # STEP 3: Director Agent (Concept Creation)
                yield _sse_format({"type": "step", "message": "Creating concept…"})
                
                # Pass url_mappings to agent so cloud URLs are injected BEFORE LLM runs
                print(f"🔍 DEBUG: Passing url_mappings to agent: {filename_to_url}")
                # 3a. Run Director stage FIRST so we can stream content immediately
                director_result = agent.director_stage(prompt, local_inputs, url_mappings=filename_to_url)
                if not isinstance(director_result, dict) or director_result.get("status") == "error":
                    msg = director_result.get("message") if isinstance(director_result, dict) else "Director stage failed"
                    yield _sse_format({"type": "error", "message": msg or "Director stage failed"})
                    return

                # Emit director_complete with content (typing effect on client)
                content_text = str(director_result.get("content") or "")
                print(f"🔍 DEBUG: director_result keys: {list(director_result.keys())}")
                print(f"🔍 DEBUG: raw content from director: {repr(director_result.get('content'))}")
                if content_text:
                    print(f"📤 Sending director_complete event with content: {content_text[:120]}...")
                    yield _sse_format({"type": "director_complete", "content": content_text})

                # 3b. Continue pipeline (Editor + Render)
                yield _sse_format({"type": "step", "message": "Generating edit…"})
                continue_result = agent.continue_after_director(
                    content=content_text,
                    abstract_plan=director_result.get("abstract_plan"),
                    analyzed_data=director_result.get("analyses", {}),
                    text_prompt=prompt,
                    media_files=local_inputs,
                    url_mappings=filename_to_url,
                )
                if not isinstance(continue_result, dict):
                    yield _sse_format({"type": "error", "message": "Unexpected agent response"})
                    return

                # Map asset srcs to original URLs if present
                try:
                    jp = continue_result.get("json_plan")
                    if isinstance(jp, dict) and filename_to_url:
                        tl = jp.get("timeline", {})
                        tracks = tl.get("tracks") or tl.get("clips")
                        def map_clip(c):
                            a = c.get("asset") if isinstance(c, dict) else None
                            if isinstance(a, dict):
                                src = a.get("src")
                                if isinstance(src, str):
                                    fname = Path(src).name
                                    if fname in filename_to_url:
                                        a["src"] = filename_to_url[fname]
                                        c["asset"] = a
                        if isinstance(tracks, list):
                            for t in tracks:
                                clips = t.get("clips", []) if isinstance(t, dict) else []
                                for c in clips:
                                    if isinstance(c, dict):
                                        map_clip(c)
                    # replace back into result
                    continue_result["json_plan"] = jp
                except Exception:
                    pass

                # Generate a plan_id and persist session snapshot for refinement
                import time as _time, uuid as _uuid
                now_ts = int(_time.time())
                plan_id = continue_result.get("plan_id") or f"plan_{now_ts}_{_uuid.uuid4().hex[:8]}"
                try:
                    SESSIONS[plan_id] = {
                        "content": content_text,
                        "abstract_plan": director_result.get("abstract_plan"),
                        "analyses": director_result.get("analyses", {}),
                        "json_plan": continue_result.get("json_plan"),
                        "media_map": filename_to_url,
                        "created_at": now_ts,
                    }
                except Exception:
                    pass

                # Attach plan_id to payload
                payload = dict(continue_result)
                payload["plan_id"] = plan_id

                # Send full result (content, analyses, ids)
                yield _sse_format({"type": "result", "payload": payload})
            finally:
                # Cleanup temps
                for p in temp_files:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

            # If render id present, stream progress until done
            render_id = continue_result.get("render_id") or (continue_result.get("rendered_video") or {}).get("render_id")
            immediate_url = continue_result.get("video_url") or (continue_result.get("rendered_video") or {}).get("video_url")
            if immediate_url:
                yield _sse_format({"type": "done", "video_url": immediate_url})
                return

            if render_id:
                tries = 0
                while tries < 180:  # up to ~9 minutes at 3s interval
                    status = agent.check_render_status(render_id)
                    # forward status/progress
                    yield _sse_format({
                        "type": "render",
                        "status": status.get("status"),
                        "progress": status.get("progress"),
                        "message": status.get("message"),
                    })
                    if status.get("video_url"):
                        yield _sse_format({"type": "done", "video_url": status["video_url"]})
                        return
                    if str(status.get("status")) in {"error", "validation_error"}:
                        yield _sse_format({"type": "error", "message": status.get("message", "Render failed")})
                        return
                    import time as _t
                    _t.sleep(3)
                    tries += 1

                yield _sse_format({"type": "error", "message": "Render timeout"})
                return

            # No render info; end stream
            yield _sse_format({"type": "done"})
        except Exception as e:
            yield _sse_format({"type": "error", "message": str(e)})
        finally:
            try:
                # Restore original analyser
                agent.media_analyser = original_analyser
            except Exception:
                pass

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/stream/refine")
def stream_refine(plan_id: str, prompt: str, new_media: Optional[str] = None, new_media_urls: Optional[str] = None):
    """SSE stream for refinement + optional rendering.
    Query params:
      - plan_id
      - prompt (refinement)
      - new_media: optional, '|' separated absolute paths
    """
    new_media_files = [m for m in (new_media or "").split("|") if m]
    new_media_urls_list = [u for u in (new_media_urls or "").split("|") if u]

    def event_gen():
        try:
            if new_media_files or new_media_urls_list:
                yield _sse_format({"type": "step", "message": "Analyzing new media…"})
                yield _sse_format({"type": "step", "message": "Planning refinement…"})
                temp_files = []
                local_inputs = list(new_media_files)
                filename_to_url = {}
                try:
                    if new_media_urls_list:
                        for u in new_media_urls_list:
                            _validate_cloud_url(u)
                            r = requests.get(u, stream=True, timeout=240)
                            if r.status_code != 200:
                                continue
                            suffix = Path(u).suffix or ".bin"
                            tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                            with tf as f:
                                shutil.copyfileobj(r.raw, f)
                            p = Path(tf.name)
                            temp_files.append(p)
                            local_inputs.append(str(p))
                            filename_to_url[p.name] = u
                    # Reuse refine_with_new_media path with url_mappings
                    print(f"🔍 DEBUG: Passing url_mappings to refine: {filename_to_url}")
                    # Prefer agent's refine_with_new_media if available
                    result = agent.refine_with_new_media(plan_id, prompt, local_inputs, url_mappings=filename_to_url)
                    
                    # Map assets to URLs
                    try:
                        jp = result.get("json_plan")
                        if isinstance(jp, dict) and filename_to_url:
                            tl = jp.get("timeline", {})
                            tracks = tl.get("tracks") or tl.get("clips")
                            def map_clip(c):
                                a = c.get("asset") if isinstance(c, dict) else None
                                if isinstance(a, dict):
                                    src = a.get("src")
                                    if isinstance(src, str):
                                        fname = Path(src).name
                                        if fname in filename_to_url:
                                            a["src"] = filename_to_url[fname]
                                            c["asset"] = a
                            if isinstance(tracks, list):
                                for t in tracks:
                                    clips = t.get("clips", []) if isinstance(t, dict) else []
                                    for c in clips:
                                        if isinstance(c, dict):
                                            map_clip(c)
                        result["json_plan"] = jp
                    except Exception:
                        pass
                finally:
                    for p in temp_files:
                        try: p.unlink(missing_ok=True)
                        except Exception: pass
            else:
                yield _sse_format({"type": "step", "message": "Refining…"})
                # Fallback: agent-driven refine without new media
                result = agent.refine_plan(plan_id, prompt)

            if not isinstance(result, dict):
                yield _sse_format({"type": "error", "message": "Unexpected agent response"})
                return

            # Update session snapshot if new json_plan or abstract_plan returned
            try:
                if isinstance(result, dict):
                    sess = SESSIONS.get(plan_id, {})
                    if result.get("json_plan"):
                        sess["json_plan"] = result.get("json_plan")
                    if result.get("abstract_plan"):
                        sess["abstract_plan"] = result.get("abstract_plan")
                    if result.get("analyses"):
                        sess["analyses"] = result.get("analyses")
                    SESSIONS[plan_id] = sess
            except Exception:
                pass

            # Attach plan_id consistently
            result_payload = dict(result) if isinstance(result, dict) else {"result": result}
            result_payload.setdefault("plan_id", plan_id)
            yield _sse_format({"type": "result", "payload": result_payload})

            render_id = result.get("render_id") or (result.get("rendered_video") or {}).get("render_id")
            immediate_url = result.get("video_url") or (result.get("rendered_video") or {}).get("video_url")
            if immediate_url:
                yield _sse_format({"type": "done", "video_url": immediate_url})
                return

            if render_id:
                tries = 0
                while tries < 180:
                    status = agent.check_render_status(render_id)
                    yield _sse_format({
                        "type": "render",
                        "status": status.get("status"),
                        "progress": status.get("progress"),
                        "message": status.get("message"),
                    })
                    if status.get("video_url"):
                        yield _sse_format({"type": "done", "video_url": status["video_url"]})
                        return
                    if str(status.get("status")) in {"error", "validation_error"}:
                        yield _sse_format({"type": "error", "message": status.get("message", "Render failed")})
                        return
                    import time as _t
                    _t.sleep(3)
                    tries += 1

                yield _sse_format({"type": "error", "message": "Render timeout"})
                return

            yield _sse_format({"type": "done"})
        except Exception as e:
            yield _sse_format({"type": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _safe_name(name: str) -> str:
    return os.path.basename(name or "upload")


@app.post("/media/upload")
async def upload_media(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept a single file upload and save it under ./uploads/. Returns path and URL."""
    try:
        fname = _safe_name(file.filename)
        dest = UPLOAD_DIR / fname
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        url = f"/uploads/{fname}"
        return {"status": "success", "path": str(dest.resolve()), "url": url, "name": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeAndCacheRequest(BaseModel):
    file_path: str = Field(..., description="Local file path to analyze")
    cloud_url: str = Field(..., description="Cloudinary URL for this file")


@app.post("/media/analyze_and_cache")
def analyze_and_cache(req: AnalyzeAndCacheRequest) -> Dict[str, Any]:
    """
    Analyze media file ONCE after upload and cache with cloud_url.
    This prevents double analysis when creating video.
    
    Usage:
    1. Upload to Cloudinary → get cloud_url
    2. Download to temp → get file_path
    3. Call this endpoint to analyze and cache
    4. Later, POST /plan will reuse cached analysis (no re-analysis!)
    """
    try:
        from pathlib import Path
        
        file_path = req.file_path
        cloud_url = req.cloud_url
        
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        print(f"🔍 Analyzing and caching: {Path(file_path).name}")
        print(f"🔗 Cloud URL: {cloud_url[:60]}...")
        
        # Analyze the file
        analyzed_data = agent.media_analyser.analyze([file_path])
        
        if not analyzed_data:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        # Get the analysis result (dict has one entry)
        filename = list(analyzed_data.keys())[0]
        analysis = analyzed_data[filename]
        
        # Inject cloud_url and set as file_path (so LLM sees it directly!)
        if isinstance(analysis, dict):
            analysis["cloud_url"] = cloud_url
            analysis["file_path"] = cloud_url  # Use cloud URL as primary path
            print(f"✅ Set file_path to cloud_url")
        
        # Store to memory with cloud_url as file_path
        from video_editing_agent import MediaAnalysis
        import datetime
        
        media_analysis = MediaAnalysis(
            file_path=cloud_url,  # Store cloud URL as file_path!
            file_type=analysis.get("file_type", "unknown"),
            filename=filename,
            analysis=analysis.get("analysis", ""),
            metadata=analysis.get("metadata", {}),
            timestamp=datetime.datetime.now().isoformat(),
            status=analysis.get("status", "success"),
            cloud_url=cloud_url
        )
        
        agent.memory.store_analysis(media_analysis)
        print(f"💾 Cached: {filename} → {cloud_url[:60]}...")
        
        return {
            "status": "success",
            "message": "Media analyzed and cached",
            "filename": filename,
            "file_type": analysis.get("file_type"),
            "cloud_url": cloud_url,
            "analysis_status": analysis.get("status", "success")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ analyze_and_cache error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/renders/{render_id}")
def get_render_status(render_id: str) -> Dict[str, Any]:
    """Expose render status so the web UI can auto-poll until the video is ready."""
    try:
        return agent.check_render_status(render_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class RefineRequest(BaseModel):
    plan_id: str = Field(..., description="Existing plan to refine")
    refinement_prompt: str = Field(..., description="Instruction describing the refinement desired")
    new_media_files: Optional[List[str]] = Field(default=None, description="Optional: additional media to analyze and integrate")
    new_media_urls: Optional[List[str]] = Field(default=None, description="Cloud URLs corresponding to new_media_files (same order)")


@app.post("/refine")
def refine_plan(req: RefineRequest) -> Dict[str, Any]:
    """Refine an existing plan using the agent's context-aware refinement.
    If new_media_files are provided, it will analyze and integrate them using
    refine_with_new_media; otherwise, it will call refine_plan.
    """
    try:
        if req.new_media_files:
            # Build url_mappings for new media if provided
            url_mappings = {}
            if req.new_media_urls and len(req.new_media_files) == len(req.new_media_urls):
                for file_path, cloud_url in zip(req.new_media_files, req.new_media_urls):
                    if cloud_url:
                        filename = Path(file_path).name
                        url_mappings[filename] = cloud_url
                        print(f"🔗 REFINE API: Mapping {filename} → {cloud_url[:60]}...")
            
            result = agent.refine_with_new_media(
                req.plan_id, 
                req.refinement_prompt, 
                req.new_media_files,
                url_mappings=url_mappings if url_mappings else None
            )
        else:
            result = agent.refine_plan(req.plan_id, req.refinement_prompt)

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Unexpected agent response")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
