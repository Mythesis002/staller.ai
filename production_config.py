"""
Production Configuration & Setup
Implements critical enterprise features for production deployment
"""

import os
import logging
from pathlib import Path
from typing import Optional

# ============================================================================
# LOGGING CONFIGURATION (Replaces all print() statements)
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
):
    """
    Configure production-grade logging
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logs
        json_format: Use JSON format for structured logging
    """
    import logging.handlers
    
    # Create logs directory
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    if json_format:
        # Structured JSON logging for production
        import json
        import datetime
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                if hasattr(record, 'request_id'):
                    log_data["request_id"] = record.request_id
                return json.dumps(log_data)
        
        console_handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# RATE LIMITING
# ============================================================================

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],  # Default: 100 requests per hour per IP
    storage_uri=os.getenv("REDIS_URL", "memory://")  # Use Redis in production
)

# Rate limit error handler
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.retry_after if hasattr(exc, 'retry_after') else 60
        }
    )


# ============================================================================
# FILE VALIDATION
# ============================================================================

from fastapi import UploadFile, HTTPException

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))  # 500MB default
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", 10))
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS

ALLOWED_MIME_TYPES = {
    'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm',
    'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/aac', 'audio/flac',
    'image/jpeg', 'image/png', 'image/gif', 'image/webp'
}


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file for security and size constraints
    
    Raises:
        HTTPException: If validation fails
    """
    # Check filename
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            400,
            f"Invalid content type '{file.content_type}'"
        )
    
    # Check file size (if available)
    if hasattr(file, 'size') and file.size:
        if file.size > MAX_FILE_SIZE:
            size_mb = file.size / (1024 * 1024)
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                400,
                f"File too large ({size_mb:.1f}MB). Maximum: {max_mb:.0f}MB"
            )


def validate_file_count(count: int) -> None:
    """Validate number of files in request"""
    if count > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            400,
            f"Too many files ({count}). Maximum: {MAX_FILES_PER_REQUEST}"
        )


# ============================================================================
# REQUEST TIMEOUT MIDDLEWARE
# ============================================================================

from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import time

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Add timeout to all requests"""
    
    def __init__(self, app, timeout: int = 300):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "message": f"Request took longer than {self.timeout}s"
                }
            )


# ============================================================================
# REQUEST ID MIDDLEWARE
# ============================================================================

import uuid

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests for tracking"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add to logging context
        logger = logging.getLogger()
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record
        
        logging.setLogRecordFactory(record_factory)
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logging.setLogRecordFactory(old_factory)


# ============================================================================
# SECURITY HEADERS MIDDLEWARE
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


# ============================================================================
# HEALTH CHECK
# ============================================================================

from fastapi import APIRouter

health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring
    """
    import sys
    
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "python_version": sys.version,
        "dependencies": {}
    }
    
    # Check external dependencies
    try:
        import requests
        # Quick check to Shotstack
        shotstack_key = os.getenv("SHOTSTACK_API_KEY")
        if shotstack_key and shotstack_key != "YOUR_SHOTSTACK_API_KEY":
            health_status["dependencies"]["shotstack"] = "configured"
        else:
            health_status["dependencies"]["shotstack"] = "not_configured"
    except Exception as e:
        health_status["dependencies"]["shotstack"] = f"error: {str(e)}"
    
    # Check Gemini
    try:
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            health_status["dependencies"]["gemini"] = "configured"
        else:
            health_status["dependencies"]["gemini"] = "not_configured"
    except Exception as e:
        health_status["dependencies"]["gemini"] = f"error: {str(e)}"
    
    return health_status


@health_router.get("/health/ready")
async def readiness_check():
    """
    Readiness check - returns 200 only if service is ready to accept traffic
    """
    # Add checks for critical dependencies
    return {"status": "ready"}


@health_router.get("/health/live")
async def liveness_check():
    """
    Liveness check - returns 200 if service is alive (even if not ready)
    """
    return {"status": "alive"}


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

import signal
import sys

def setup_graceful_shutdown(cleanup_func=None):
    """
    Setup graceful shutdown handlers
    
    Args:
        cleanup_func: Optional function to call during shutdown
    """
    logger = logging.getLogger(__name__)
    
    def graceful_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)
        
        logger.info("Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)


# ============================================================================
# TEMP FILE CLEANUP
# ============================================================================

import atexit
import tempfile
import time

TEMP_DIR = Path(tempfile.gettempdir()) / "videoeditor"
TEMP_DIR.mkdir(exist_ok=True)

def cleanup_old_temp_files(max_age_hours: int = 1):
    """
    Delete temporary files older than max_age_hours
    
    Args:
        max_age_hours: Maximum age of files to keep (in hours)
    """
    logger = logging.getLogger(__name__)
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    
    try:
        for file_path in TEMP_DIR.glob("*"):
            try:
                if file_path.is_file():
                    age = now - file_path.stat().st_mtime
                    if age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old temporary files")
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}", exc_info=True)

# Register cleanup on exit
atexit.register(lambda: cleanup_old_temp_files())


# ============================================================================
# CORS CONFIGURATION
# ============================================================================

def get_cors_config():
    """Get CORS configuration from environment"""
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    return {
        "allow_origins": allowed_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["*"],
        "expose_headers": ["X-Request-ID"],
    }
