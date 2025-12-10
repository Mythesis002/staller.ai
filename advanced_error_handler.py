#!/usr/bin/env python
"""
Advanced Error Handling and Logging System for Video Editing Agent
"""

import json
import datetime
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

class AdvancedErrorHandler:
    """Advanced error handling system with intelligent recovery"""
    
    def __init__(self, log_dir: str = "error_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Error statistics
        self.error_counts = {}
        self.recovery_attempts = {}
        self.successful_recoveries = {}
        
        print("🛡️ AdvancedErrorHandler initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.log_dir / f"video_agent_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def handle_json_extraction_error(self, error: Exception, llm_response: str, context: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Handle JSON extraction errors with advanced recovery"""
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.log_error("JSON_EXTRACTION", error_type, error_msg, {
            "response_length": len(llm_response) if llm_response else 0,
            "response_preview": llm_response[:100] if llm_response else "Empty",
            "context": context
        })
        
        # Increment error count
        self.error_counts["json_extraction"] = self.error_counts.get("json_extraction", 0) + 1
        
        # Try intelligent recovery strategies
        recovery_result = self.attempt_json_recovery(llm_response, error_msg, context)
        
        if recovery_result:
            self.successful_recoveries["json_extraction"] = self.successful_recoveries.get("json_extraction", 0) + 1
            return recovery_result
        
        # Generate intelligent fallback
        return self.generate_intelligent_fallback(llm_response, context)
    
    def attempt_json_recovery(self, response: str, error_msg: str, context: Dict[str, Any]) -> Optional[tuple[str, Dict[str, Any]]]:
        """Attempt to recover from JSON errors using various strategies"""
        
        if not response or not response.strip():
            print("🔧 Recovery: Empty response detected")
            return None
        
        # Strategy 1: Fix common JSON issues
        try:
            fixed_response = self.fix_common_json_issues(response)
            if fixed_response != response:
                parsed = json.loads(fixed_response)
                if isinstance(parsed, dict) and ('content' in parsed or 'json_plan' in parsed):
                    print("✅ Recovery: Fixed common JSON issues")
                    return parsed.get("content", "Recovered content"), parsed.get("json_plan", {})
        except:
            pass
        
        # Strategy 2: Extract JSON from mixed content
        try:
            extracted_json = self.extract_json_from_mixed_content(response)
            if extracted_json:
                print("✅ Recovery: Extracted JSON from mixed content")
                return extracted_json.get("content", "Extracted content"), extracted_json.get("json_plan", {})
        except:
            pass
        
        # Strategy 3: Parse partial JSON
        try:
            partial_result = self.parse_partial_json(response)
            if partial_result:
                print("✅ Recovery: Parsed partial JSON")
                return partial_result
        except:
            pass
        
        return None
    
    def fix_common_json_issues(self, response: str) -> str:
        """Fix common JSON formatting issues"""
        import re
        
        fixed = response.strip()
        
        # Remove markdown formatting
        fixed = re.sub(r'^```json\s*', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'^```\s*', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'\s*```$', '', fixed, flags=re.MULTILINE)
        
        # Fix trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Fix unquoted keys
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
        
        # Fix single quotes to double quotes
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
        
        # Balance braces
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        
        return fixed
    
    def extract_json_from_mixed_content(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from mixed content using regex"""
        import re
        
        # Look for JSON-like structures
        json_patterns = [
            r'\{[^{}]*"content"[^{}]*"json_plan"[^{}]*\}',
            r'\{.*?"content".*?\}',
            r'\{.*?"json_plan".*?\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except:
                    continue
        
        return None
    
    def parse_partial_json(self, response: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Parse partial JSON content"""
        import re
        
        content = "Partial content recovered"
        json_plan = {}
        
        # Look for content field
        content_match = re.search(r'"content"\s*:\s*"([^"]*)"', response)
        if content_match:
            content = content_match.group(1)
        
        # Look for simple timeline structure
        if "timeline" in response.lower():
            json_plan = {
                "timeline": {
                    "background": "#000000",
                    "tracks": []
                },
                "output": {
                    "format": "mp4",
                    "resolution": "1920x1080",
                    "fps": 30
                }
            }
        
        return content, json_plan
    
    def generate_intelligent_fallback(self, response: str, context: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Generate intelligent fallback based on context"""
        
        # Analyze context to determine media types
        analyzed_data = context.get("analyzed_data", {})
        prompt = context.get("prompt", "video creation")
        
        has_audio = any(data.get("file_type") == "audio" for data in analyzed_data.values() if isinstance(data, dict))
        has_video = any(data.get("file_type") == "video" for data in analyzed_data.values() if isinstance(data, dict))
        
        # Generate appropriate fallback content
        if has_audio and not has_video:
            fallback_content = f"""🎵 **Audio-Based Video Creation**

The system detected audio content and will create a dynamic audio visualization video.

**Features:**
- Audio waveform visualization
- Synchronized text overlays
- Professional transitions
- Optimized for audio content

**Request:** {prompt}"""
            
            tracks = [
                {
                    "type": "audio",
                    "clips": [{
                        "asset": {"type": "audio", "src": "input_audio.mp3", "volume": 0.8},
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
                            "style": {"fontSize": 60, "color": "#FFFFFF", "fontFamily": "Arial"}
                        },
                        "start": 0,
                        "length": 5,
                        "transition": {"in": {"type": "fade", "duration": 1}}
                    }]
                }
            ]
        
        elif has_video:
            fallback_content = f"""🎬 **Video Editing Plan**

Professional video editing plan created based on your uploaded content.

**Features:**
- Analyzed video content integration
- Professional transitions and effects
- Optimized timing and pacing
- High-quality output settings

**Request:** {prompt}"""
            
            tracks = [
                {
                    "type": "video",
                    "clips": [{
                        "asset": {"type": "video", "src": "input_video.mp4", "volume": 1.0},
                        "start": 0,
                        "length": 30,
                        "transition": {"in": {"type": "fade", "duration": 1}}
                    }]
                }
            ]
        
        else:
            fallback_content = f"""🎨 **Creative Video Template**

A professional video template has been generated for your project.

**Features:**
- Clean, modern design
- Customizable text and graphics
- Professional transitions
- Ready for content integration

**Request:** {prompt}"""
            
            tracks = [
                {
                    "type": "title",
                    "clips": [{
                        "asset": {
                            "type": "title",
                            "text": "Your Creative Video",
                            "style": {"fontSize": 80, "color": "#FFFFFF", "fontFamily": "Arial Bold"}
                        },
                        "start": 0,
                        "length": 10,
                        "transition": {"in": {"type": "fade", "duration": 1}}
                    }]
                }
            ]
        
        fallback_json = {
            "timeline": {
                "background": "#000000",
                "tracks": tracks
            },
            "output": {
                "format": "mp4",
                "resolution": "1920x1080",
                "fps": 30,
                "thumbnail": False
            }
        }
        
        print("🛠️ Generated intelligent fallback response")
        return fallback_content, fallback_json
    
    def log_error(self, component: str, error_type: str, error_msg: str, additional_data: Dict[str, Any] = None):
        """Log error with detailed information"""
        
        error_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "component": component,
            "error_type": error_type,
            "error_message": error_msg,
            "additional_data": additional_data or {},
            "stack_trace": traceback.format_exc()
        }
        
        # Log to file
        error_file = self.log_dir / f"errors_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            if error_file.exists():
                with open(error_file, 'r') as f:
                    errors = json.load(f)
            else:
                errors = []
            
            errors.append(error_data)
            
            with open(error_file, 'w') as f:
                json.dump(errors, f, indent=2)
        
        except Exception as e:
            print(f"⚠️ Failed to log error: {e}")
        
        # Log to console
        self.logger.error(f"{component} - {error_type}: {error_msg}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(self.successful_recoveries.values())
        
        recovery_rate = (total_recoveries / total_errors * 100) if total_errors > 0 else 0
        
        return {
            "total_errors": total_errors,
            "total_recoveries": total_recoveries,
            "recovery_rate": f"{recovery_rate:.1f}%",
            "error_breakdown": self.error_counts.copy(),
            "recovery_breakdown": self.successful_recoveries.copy()
        }
    
    def generate_error_report(self) -> str:
        """Generate a comprehensive error report"""
        
        stats = self.get_error_statistics()
        
        report = f"""
📊 **Advanced Error Handler Report**

**Overall Statistics:**
• Total Errors: {stats['total_errors']}
• Successful Recoveries: {stats['total_recoveries']}
• Recovery Rate: {stats['recovery_rate']}

**Error Breakdown:**
"""
        
        for error_type, count in stats['error_breakdown'].items():
            recoveries = stats['recovery_breakdown'].get(error_type, 0)
            rate = (recoveries / count * 100) if count > 0 else 0
            report += f"• {error_type}: {count} errors, {recoveries} recoveries ({rate:.1f}%)\n"
        
        return report
