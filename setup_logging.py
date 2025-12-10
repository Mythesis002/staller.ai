#!/usr/bin/env python
"""
Global logging configuration to reduce excessive API call logging
"""

import logging
import warnings
from config import ENABLE_VERBOSE_LOGGING, ENABLE_GRADIO_LOGGING

def setup_optimized_logging():
    """Configure logging to reduce excessive output from API calls"""
    
    if not ENABLE_VERBOSE_LOGGING:
        # Reduce HTTP request logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        # Suppress urllib3 warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
        
    if not ENABLE_GRADIO_LOGGING:
        # Reduce Gradio client logging
        logging.getLogger("gradio_client").setLevel(logging.ERROR)
        logging.getLogger("gradio").setLevel(logging.ERROR)
        
        # Suppress specific gradio warnings
        warnings.filterwarnings("ignore", message=".*gradio.*")
    
    # Set root logger to INFO level to maintain important messages
    logging.getLogger().setLevel(logging.INFO)
    
    print("🔇 Optimized logging configured - reduced API call verbosity")

def restore_verbose_logging():
    """Restore verbose logging for debugging"""
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.INFO)
    logging.getLogger("gradio_client").setLevel(logging.INFO)
    logging.getLogger("gradio").setLevel(logging.INFO)
    
    print("🔊 Verbose logging restored")

# Auto-setup on import
setup_optimized_logging()
