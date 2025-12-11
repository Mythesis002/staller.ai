#!/usr/bin/env python
"""
PRODUCTION-READY FastAPI Server
Implements enterprise features: logging, rate limiting, validation, monitoring
"""

import os
import logging
from typing import List, Optional, Any, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import requests
import tempfile
import shutil

# Import production configuration
from production_config import (
    setup_logging,
    limiter,
    rate_limit_handler,
    validate_file,
    validate_file_count,
    TimeoutMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    health_router,
    setup_graceful_shutdown,
    cleanup_old_temp_files,
    get_cors_config,
)

# Setup logging FIRST (before any other imports that might log)
logger = setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/api.log"),
    json_format=os.getenv("LOG_FORMAT", "json") == "json"
)

# Now import application code
from main_video_agent import VideoEditingAgent

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Video Editing Agent API",
    version="1.0.0",
    description="Enterprise-grade video editing automation API",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "1") == "1" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "1") == "1" else None,
)

# ============================================================================
# MIDDLEWARE (Order matters!)
# ============================================================================



# 2. CORS
cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# 3. Security Headers
app.add_middleware(SecurityHeadersMiddleware)

# 4. Request ID Tracking
app.add_middleware(RequestIDMiddleware)

# 5. Timeout Protection
app.add_middleware(
    TimeoutMiddleware,
    timeout=int(os.getenv("REQUEST_TIMEOUT", "300"))
)

# 6. Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(Exception, rate_limit_handler)

# ============================================================================
# DIRECTORIES & STATIC FILES
# ============================================================================

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

# ============================================================================
# AGENT & CACHE INITIALIZATION
# ============================================================================

logger.info("Initializing Video Editing Agent...")
agent = VideoEditingAgent()

# In-memory cache (use Redis in production)
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
from threading import Lock
_ANALYSIS_LOCK = Lock()
ANALYSIS_INFLIGHT = set()

# Cached analyser wrapper
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

    def analyze(self, path: str):
        cached = self._get_cached(path)
        if cached is not None:
            return cached
        if hasattr(self.inner, 'analyze') and callable(getattr(self.inner, 'analyze')):
            res = self.inner.analyze(path)
            if isinstance(res, dict):
                self._store(path, res)
            return res
        # Fallback routing
        ext = Path(path).suffix.lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']:
            return self.analyze_video(path)
        if ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']:
            return self.analyze_audio(path)
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return self.analyze_image(path)
        return {"status": "skipped", "message": f"Unsupported type {ext or 'unknown'}", "cached": False}


try:
    agent.media_analyser = CachedAnalyser(agent.media_analyser, ANALYSIS_CACHE)
    logger.info("Cached analyser initialized")
except Exception as e:
    logger.warning(f"Could not wrap analyser with cache: {e}")

# ============================================================================
# HEALTH CHECKS
# ============================================================================

app.include_router(health_router)

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the web UI"""
    logger.info("Serving index page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse(content="", status_code=204)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PlanRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="User instruction")
    media_files: Optional[List[str]] = Field(default_factory=list, description="Local file paths")
    media_urls: Optional[List[str]] = Field(default_factory=list, description="Cloud URLs")


class AnalyzeRequest(BaseModel):
    media_files: List[str] = Field(..., description="Absolute paths to local media files")


class AnalyzeRemoteRequest(BaseModel):
    url: str = Field(..., description="HTTPS URL to media")
    filename: Optional[str] = Field(default=None, description="Optional filename hint")


class CloudinaryUploadRequest(BaseModel):
    path: str = Field(..., description="Absolute path to local file")


class AnalyzeAndCacheRequest(BaseModel):
    file_path: str = Field(..., description="Local file path")
    cloud_url: str = Field(..., description="Cloudinary URL")

# ============================================================================
# HELPERS
# ============================================================================

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

    # Explicit allowlist
    if allowed_hosts and any(host.endswith(h) or host == h for h in allowed_hosts):
        return

    # Accept Cloudinary default domain with correct cloud name in path: /<cloud_name>/...
    if host.endswith("res.cloudinary.com"):
        if not allowed_cloud:
            return
        if path.startswith(f"/{allowed_cloud}/"):
            return
        raise HTTPException(status_code=400, detail="Cloudinary URL cloud name mismatch")

    # Fallback: if CLOUDINARY_CLOUD_NAME is set and present in host (custom domains)
    if allowed_cloud and allowed_cloud in host:
        return

    raise HTTPException(status_code=400, detail="URL host not allowed")


def _sse_format(data: Dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    import json as _json
    return f"data: {_json.dumps(data)}\n\n"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/plan")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute
async def create_plan(req: PlanRequest, request: Request) -> Dict[str, Any]:
    """Create a new editing plan"""
    logger.info(
        "Creating plan",
        extra={
            "prompt_length": len(req.prompt),
            "media_count": len(req.media_files or [])
        }
    )
    
    try:
        # Validate file count
        if req.media_files:
            validate_file_count(len(req.media_files))
        
        # Build url_mappings
        url_mappings = {}
        if req.media_files and req.media_urls and len(req.media_files) == len(req.media_urls):
            for file_path, cloud_url in zip(req.media_files, req.media_urls):
                if cloud_url:
                    filename = Path(file_path).name
                    url_mappings[filename] = cloud_url
                    logger.debug(f"Mapping {filename} → {cloud_url[:60]}...")
        
        result = agent.process_request(
            req.prompt,
            req.media_files or [],
            url_mappings=url_mappings if url_mappings else None
        )
        
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Unexpected agent response")
        
        logger.info("Plan created successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Plan creation failed: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/media/cloudinary/sign")
async def cloudinary_sign() -> Dict[str, Any]:
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
        params = {'timestamp': str(timestamp), 'folder': folder}
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


@app.post("/media/analyze_remote")
async def analyze_remote(req: AnalyzeRemoteRequest) -> Dict[str, Any]:
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
                raise HTTPException(
                    status_code=400,
                    detail=f"Download failed: {r.status_code} "
                           f"{r.text[:200] if hasattr(r, 'text') else ''}",
                )
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
            res_dict = (
                res if isinstance(res, dict)
                else getattr(res, '__dict__', {"status": "ok", "analysis": str(res)})
            )
            ANALYSIS_CACHE[key] = res_dict

            # Persist to agent memory
            try:
                from video_editing_agent import MediaAnalysis
                import datetime

                filename = req.filename or Path(req.url).name

                media_analysis = MediaAnalysis(
                    file_path=req.url,
                    file_type=res_dict.get("file_type", "unknown"),
                    filename=filename,
                    analysis=res_dict.get("analysis", ""),
                    metadata=res_dict.get("metadata", {}),
                    timestamp=datetime.datetime.now().isoformat(),
                    status=res_dict.get("status", "success"),
                    cloud_url=req.url,
                )

                agent.memory.store_analysis(media_analysis)
                logger.info(f"Stored analysis for {filename} → {req.url[:60]}...")
            except Exception as persist_err:
                logger.warning(f"Failed to persist analysis: {persist_err}")

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


@app.get("/stream/plan")
def stream_plan(
    prompt: str,
    media: Optional[str] = None,
    skip_media: int = 0,
    media_urls: Optional[str] = None,
):
    """SSE stream for planning + optional rendering."""
    media_files = [m for m in (media or "").split("|") if m]
    media_urls_list = [u for u in (media_urls or "").split("|") if u]

    def event_gen():
        try:
            original_analyser = agent.media_analyser

            # If skip_media and no new URLs, we conceptually would use CachedOnlyAnalyser;
            # for now, just keep original_analyser (remote-only flow mainly uses media_urls).
            temp_files = []
            filename_to_url = {}
            try:
                local_inputs = list(media_files)

                # STEP 1: Media Analysis (remote URLs -> temp files)
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

                director_result = agent.director_stage(
                    prompt,
                    local_inputs,
                    url_mappings=filename_to_url,
                )
                if not isinstance(director_result, dict) or director_result.get("status") == "error":
                    msg = director_result.get("message") if isinstance(director_result, dict) else "Director stage failed"
                    yield _sse_format({"type": "error", "message": msg or "Director stage failed"})
                    return

                # Emit director_complete with content
                content_text = str(director_result.get("content") or "")
                if content_text:
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
                    continue_result["json_plan"] = jp
                except Exception:
                    pass

                # Generate a plan_id
                import time as _time, uuid as _uuid
                now_ts = int(_time.time())
                plan_id = continue_result.get("plan_id") or f"plan_{now_ts}_{_uuid.uuid4().hex[:8]}"

                payload = dict(continue_result)
                payload["plan_id"] = plan_id

                # Send full result
                yield _sse_format({"type": "result", "payload": payload})

                # Render status handling (if any)
                render_id = continue_result.get("render_id") or (continue_result.get("rendered_video") or {}).get("render_id")
                immediate_url = continue_result.get("video_url") or (continue_result.get("rendered_video") or {}).get("video_url")

                if immediate_url:
                    yield _sse_format({"type": "done", "video_url": immediate_url})
                    return

                if render_id:
                    tries = 0
                    while tries < 180:  # up to ~9 minutes at 3s interval
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

                # No render info; end stream
                yield _sse_format({"type": "done"})
            finally:
                # Cleanup temps
                for p in temp_files:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as e:
            yield _sse_format({"type": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Application starting up...")
    cleanup_old_temp_files()  # Clean old files on startup
    logger.info("Application ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Application shutting down...")
    cleanup_old_temp_files()  # Final cleanup
    logger.info("Application shutdown complete")


# Setup graceful shutdown handlers
setup_graceful_shutdown(cleanup_func=cleanup_old_temp_files)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server_production:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
        log_config=None,  # Use our custom logging
        access_log=False,  # We handle access logs ourselves
    )
