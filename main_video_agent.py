#!/usr/bin/env python
"""
Clean Video Editing Agent Orchestrator - Following Workflow Diagram
INPUT → MEDIA_ANALYSER → LLM → EXTRACTOR → SHOTSTACK → GUI
"""

import os
import uuid
import json
import datetime
from dataclasses import asdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from jsonschema import validate, ValidationError

# Import clean modular components
from video_editing_agent import (
    MediaAnalyser, EnhancedLLMProcessor, DirectorAgent, EditorAgent, TransitionAgent, AdvancedJSONExtractor, ShotstackRenderer,
    MediaAnalysis, EditingPlan, MemoryManager, PromptEnhancer, MusicGenerator
)
from advanced_error_handler import AdvancedErrorHandler
from config import ASSET_ENFORCEMENT_MODE


class VideoEditingAgent:
    """Clean orchestrator following workflow diagram: INPUT → MEDIA_ANALYSER → LLM → EXTRACTOR → SHOTSTACK → GUI"""
    
    def __init__(self):
        # Initialize advanced modular components
        self.media_analyser = MediaAnalyser()
        self.director_agent = DirectorAgent()
        self.transition_agent = TransitionAgent()  # NEW: Transition selection agent
        self.editor_agent = EditorAgent()
        self.llm_processor = EnhancedLLMProcessor()  # Keep for backward compatibility
        self.json_extractor = AdvancedJSONExtractor()
        self.shotstack_renderer = ShotstackRenderer()
        self.memory = MemoryManager()
        self.error_handler = AdvancedErrorHandler()
        self.music_generator = MusicGenerator()
        # New: Prompt enhancer (Gemini-based)
        try:
            self.prompt_enhancer = PromptEnhancer()
        except Exception:
            # If enhancer cannot initialize, use a simple shim
            class _PassEnhancer:
                def enhance(self, p, analyses=None):
                    return p, {"used": False}
            self.prompt_enhancer = _PassEnhancer()
        # Load timeline schema once
        schema_path = Path(__file__).parent / "schemas" / "timeline_v1.json"
        self.timeline_schema = None
        try:
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.timeline_schema = json.load(f)
        except Exception as _:
            self.timeline_schema = None
        
        print("🎬 Advanced Video Editing Agent initialized successfully!")
        
        # Agent state
        self.current_session = {
            "media_files": [],
            "analyses": {},
            "current_plan": None,
            "render_history": [],
            "conversation_context": []
        }
    
    def process_request(
        self,
        text_prompt: str,
        media_files: List[str] = None,
        url_mappings: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Clean workflow orchestration following diagram:
        INPUT → MEDIA_ANALYSER → LLM → EXTRACTOR → SHOTSTACK → GUI
        
        Args:
            text_prompt: User's editing request
            media_files: List of local file paths
            url_mappings: Optional dict mapping temp filenames to cloud URLs
        """
        
        try:
            print(f"🚀 Processing: {text_prompt[:50]}...")
            
            # STEP 1: INPUT VALIDATION
            if not text_prompt.strip():
                return self._error_response("❌ Please provide a text prompt")
            
            # STEP 2: MEDIA_ANALYSER (if media provided)
            analyzed_data = {}
            if media_files and len(media_files) > 0:
                print("📹 Step 2: Media Analysis...")
                
                # CHECK CACHE FIRST to avoid double analysis
                files_to_analyze = []
                for file_path in media_files:
                    temp_filename = Path(file_path).name
                    cached = None
                    
                    # Try URL-based lookup first (most reliable!)
                    if url_mappings and temp_filename in url_mappings:
                        cloud_url = url_mappings[temp_filename]
                        cached = self.memory.get_analysis_by_url(cloud_url)
                        if cached:
                            print(f"♻️  Using cached analysis (by URL) for: {temp_filename}")
                    
                    # Fallback to filename-based lookup
                    if not cached:
                        cached = self.memory.get_latest_analysis(temp_filename)
                        if cached:
                            print(f"♻️  Using cached analysis (by filename) for: {temp_filename}")
                    
                    if cached:
                        # Convert MediaAnalysis to dict format
                        analyzed_data[temp_filename] = {
                            "file_path": cached.file_path,
                            "file_type": cached.file_type,
                            "analysis": cached.analysis,
                            "metadata": cached.metadata,
                            "status": cached.status,
                            "cloud_url": cached.cloud_url  # Already has cloud_url!
                        }
                        print(f"✅ Cached cloud_url: {cached.cloud_url[:60] if cached.cloud_url else 'None'}...")
                    else:
                        files_to_analyze.append(file_path)
                
                # ANALYZE ONLY NEW FILES (not in cache)
                if files_to_analyze:
                    print(f"🔍 Analyzing {len(files_to_analyze)} new file(s)...")
                    new_analyzed = self.media_analyser.analyze(files_to_analyze)
                    
                    # INJECT CLOUD URLs for newly analyzed files
                    if url_mappings:
                        print("🔗 Injecting cloud URLs into newly analyzed data...")
                        for filename, analysis in new_analyzed.items():
                            if isinstance(analysis, dict):
                                file_path = analysis.get("file_path", "")
                                temp_name = Path(file_path).name if file_path else filename
                                if temp_name in url_mappings:
                                    cloud_url = url_mappings[temp_name]
                                    analysis["cloud_url"] = cloud_url
                                    # SIMPLE FIX: Store cloud URL as file_path so LLM sees it directly!
                                    analysis["file_path"] = cloud_url
                                    print(f"✅ Set file_path to cloud_url for {filename}: {cloud_url[:60]}...")
                    
                    # Merge with analyzed_data
                    analyzed_data.update(new_analyzed)
                    
                    # Store NEW analyzed data to cache
                    for filename, analysis in new_analyzed.items():
                        if isinstance(analysis, dict):
                            # Use cloud_url as file_path if available (primary identifier)
                            cloud_url = analysis.get("cloud_url", "")
                            stored_file_path = cloud_url if cloud_url else analysis.get("file_path", "")
                            
                            media_analysis = MediaAnalysis(
                                file_path=stored_file_path,  # Cloudinary URL if available!
                                file_type=analysis.get("file_type", "unknown"),
                                filename=filename,
                                analysis=analysis.get("analysis", ""),
                                metadata=analysis.get("metadata", {}),
                                timestamp=datetime.datetime.now().isoformat(),
                                status=analysis.get("status", "success"),
                                cloud_url=cloud_url
                            )
                            self.memory.store_analysis(media_analysis)
                            print(f"💾 Cached analysis: {filename} → {stored_file_path[:60] if stored_file_path else 'No path'}...")
                else:
                    print("✅ All files already cached - no analysis needed!")
            
            # STEP 3: PROMPT ENHANCEMENT (Gemini)
            enhanced_prompt, enhancer_meta = self.prompt_enhancer.enhance(text_prompt, analyzed_data)
            if enhancer_meta.get("used"):
                print("📝 PromptEnhancer used → preview:", (enhanced_prompt or "")[:120] + ("…" if enhanced_prompt and len(enhanced_prompt)>120 else ""))
            else:
                print("📝 PromptEnhancer bypassed (pass-through)")

            # STEP 4: DIRECTOR AGENT (Semantic Planning)
            print("🎭 Step 3a: Director Agent - Semantic Planning...")
            # Concise input log
            media_keys = list((analyzed_data or {}).keys())
            dp = enhanced_prompt if enhanced_prompt is not None else text_prompt
            print(f"   ▶ Director INPUT | prompt='{dp[:80]}{'…' if len(dp)>80 else ''}', media={len(media_keys)} -> {media_keys[:5]}")

            director_attempts = 0
            director_used = False
            director_fallback = False
            legacy_used = False
            try:
                director_attempts += 1
                content, abstract_plan = self.director_agent.plan(dp, analyzed_data)
                director_used = True
                # Concise output log
                style = (abstract_plan or {}).get("style")
                dur = (abstract_plan or {}).get("target_duration")
                tracks_count = len((abstract_plan or {}).get("tracks", []))
                print(f"   ◀ Director OUTPUT | content='{(content or '')[:80]}{'…' if content and len(content)>80 else ''}', style={style}, duration={dur}s, tracks={tracks_count}")
                print("✅ Director agent completed successfully")
            except Exception as e:
                print(f"❌ Director agent failed: {str(e)}")
                # Use director's own fallback first (faster)
                print("🔄 Using Director fallback response...")
                try:
                    content, abstract_plan = self.director_agent._generate_fallback_director_response(analyzed_data, text_prompt)
                    director_fallback = True
                    style = (abstract_plan or {}).get("style")
                    dur = (abstract_plan or {}).get("target_duration")
                    tracks_count = len((abstract_plan or {}).get("tracks", []))
                    print(f"   ◀ Director FALLBACK | content='{(content or '')[:80]}{'…' if content and len(content)>80 else ''}', style={style}, duration={dur}s, tracks={tracks_count}")
                    print("✅ Director fallback completed successfully")
                except Exception as fallback_e:
                    print(f"❌ Director fallback also failed: {str(fallback_e)}")
                    # Final fallback to old LLM processor
                    print("🔄 Falling back to legacy LLM processor...")
                    llm_response = self.llm_processor.process(analyzed_data, text_prompt)
                    legacy_used = True
                    if llm_response.startswith("Error:"):
                        return self._error_response(llm_response)
                    try:
                        content, json_plan = self.json_extractor.extract_and_separate(llm_response)
                    except Exception as extract_e:
                        context = {
                            "analyzed_data": analyzed_data,
                            "prompt": text_prompt,
                            "llm_response": llm_response
                        }
                        content, json_plan = self.error_handler.handle_json_extraction_error(extract_e, llm_response, context)
                    # Skip to validation step
                    abstract_plan = None
            
            # STEP 4.5: MUSIC GENERATION (optional)
            try:
                if isinstance(abstract_plan, dict):
                    music_script = abstract_plan.get("music_script")
                    if isinstance(music_script, str) and music_script.strip():
                        print("🎵 Music script detected — generating soundtrack…")
                        print(f"   Music script preview: {music_script[:100]}...")
                        music_url = self.music_generator.generate(music_script)
                        if isinstance(music_url, str) and music_url.startswith("http"):
                            # Inject into analyses and mappings as a synthetic audio asset
                            synth_name = "generated_music.mp3"
                            analyzed_data[synth_name] = {
                                "file_path": music_url,
                                "file_type": "audio",
                                "analysis": "AI-generated background music",
                                "metadata": {"source": "music_generator"},
                                "status": "success",
                                "cloud_url": music_url,
                            }
                            if url_mappings is None:
                                url_mappings = {}
                            url_mappings[synth_name] = music_url
                            print(f"✅ Music generated and injected: {music_url}")
                        else:
                            print(f"⚠️ Music generation returned no valid URL: {music_url}")
                    else:
                        print("ℹ️ No music_script in abstract_plan — skipping music generation")
                else:
                    print("ℹ️ abstract_plan is not a dict — skipping music generation")
            except Exception as me:
                print(f"⚠️ Music generation exception: {me}")
                import traceback
                traceback.print_exc()

            # STEP 4.75: TRANSITION AGENT (Select chroma key transitions)
            enriched_plan = abstract_plan  # Default: use original plan
            if abstract_plan is not None and isinstance(abstract_plan, dict):
                print("🎨 Step 4.75: Transition Agent - Selecting transitions…")
                try:
                    enriched_plan = self.transition_agent.select_transitions(abstract_plan, analyzed_data)
                    transitions_count = len(enriched_plan.get("transitions", []))
                    print(f"   ◀ Transition Agent OUTPUT | selected {transitions_count} transitions")
                    print("✅ Transition agent completed successfully")
                except Exception as te:
                    print(f"⚠️ Transition agent failed: {str(te)}")
                    print("   Continuing with original plan (no transitions)")
                    enriched_plan = abstract_plan
            else:
                print("ℹ️ Skipping Transition Agent (no abstract_plan available)")

            # STEP 5: EDITOR AGENT (Shotstack JSON Generation)
            editor_attempts = 0
            editor_used = False
            editor_fallback = False
            if enriched_plan is not None:
                print("🎬 Step 3b: Editor Agent - Shotstack JSON Generation…")
                # Concise input log for editor
                in_style = (enriched_plan or {}).get("style")
                in_dur = (enriched_plan or {}).get("target_duration")
                in_tracks = len((enriched_plan or {}).get("tracks", []))
                in_transitions = len((enriched_plan or {}).get("transitions", []))
                print(f"   ▶ Editor INPUT | style={in_style}, duration={in_dur}s, tracks={in_tracks}, transitions={in_transitions}")
                try:
                    editor_attempts += 1
                    json_plan = self.editor_agent.build_timeline(enriched_plan, analyzed_data, url_mappings)
                    editor_used = True
                    # Concise output log
                    t_tracks = len((json_plan or {}).get("timeline", {}).get("tracks", []))
                    total_clips = sum(len(t.get("clips", [])) for t in (json_plan or {}).get("timeline", {}).get("tracks", []))
                    print(f"   ◀ Editor OUTPUT | tracks={t_tracks}, clips={total_clips}")
                    print("✅ Editor agent completed successfully")
                except Exception as e:
                    print(f"❌ Editor agent failed: {str(e)}")
                    # Use editor's own fallback first (faster)
                    print("🔄 Using Editor fallback response...")
                    try:
                        json_plan = self.editor_agent._generate_fallback_timeline(enriched_plan, analyzed_data, url_mappings)
                        editor_fallback = True
                        t_tracks = len((json_plan or {}).get("timeline", {}).get("tracks", []))
                        total_clips = sum(len(t.get("clips", [])) for t in (json_plan or {}).get("timeline", {}).get("tracks", []))
                        print(f"   ◀ Editor FALLBACK | tracks={t_tracks}, clips={total_clips}")
                        print("✅ Editor fallback completed successfully")
                    except Exception as fallback_e:
                        print(f"❌ Editor fallback also failed: {str(fallback_e)}")
                        # Final fallback to old LLM processor
                        print("🔄 Falling back to legacy LLM processor...")
                        llm_response = self.llm_processor.process(analyzed_data, text_prompt)
                        legacy_used = True
                        if llm_response.startswith("Error:"):
                            return self._error_response(llm_response)
                        try:
                            content, json_plan = self.json_extractor.extract_and_separate(llm_response)
                        except Exception as extract_e:
                            context = {
                                "analyzed_data": analyzed_data,
                                "prompt": text_prompt,
                                "llm_response": llm_response
                            }
                            content, json_plan = self.error_handler.handle_json_extraction_error(extract_e, llm_response, context)
            
            # STEP 6: VALIDATION AND URL MAPPING
            try:
                if json_plan:
                    # Step 1: Validate structure (required)
                    json_plan = self._validate_and_normalize_plan(json_plan)
                    
                    # Step 2: Check if URLs are valid (smart validation)
                    needs_url_fix = self._check_if_urls_need_fixing(json_plan)
                    
                    if needs_url_fix:
                        print("⚠️ Detected invalid URLs in LLM output - applying fixes...")
                        allowed_names = set(analyzed_data.keys()) if analyzed_data else set()
                        if media_files:
                            allowed_names.update({Path(p).name for p in media_files})
                        
                        json_plan = self._enforce_allowed_assets(json_plan, allowed_names)
                        json_plan = self._map_temp_filenames_to_urls(json_plan, analyzed_data)
                        json_plan = self._force_replace_invalid_urls(json_plan, analyzed_data, url_mappings)
                        json_plan = self._validate_and_normalize_plan(json_plan)  # Re-validate
                    else:
                        print("✅ LLM generated valid cloud URLs - no URL fixing needed!")
            except Exception as e:
                print(f"⚠️ Plan validation error: {e}")

            # STEP 7: SHOTSTACK RENDERER (optional)
            render_info = None
            render_id = None
            render_status = None
            video_url = None
            if json_plan:
                # Validate plan before attempting to render
                if self.timeline_schema is not None:
                    try:
                        validate(instance=json_plan, schema=self.timeline_schema)
                    except ValidationError as ve:
                        render_info = {
                            "status": "validation_error",
                            "message": f"JSON plan failed schema validation: {ve.message}",
                            "render_id": None,
                            "video_url": None
                        }
                        render_id = None
                        render_status = "validation_error"
                        video_url = None
                    else:
                        print("🎬 Step 6: Shotstack Rendering...")
                        render_info = self.shotstack_renderer.render_video(json_plan)
                else:
                    # No schema available, proceed with best effort
                    print("🎬 Step 6: Shotstack Rendering...")
                    render_info = self.shotstack_renderer.render_video(json_plan)
                # Map to expected fields
                if isinstance(render_info, dict):
                    render_id = render_info.get("render_id")
                    render_status = render_info.get("status")
                    video_url = render_info.get("video_url")
            
            # STEP 8: PREPARE GUI RESPONSE
            agent_trace = {
                "director_used": director_used,
                "director_fallback": director_fallback,
                "director_attempts": director_attempts,
                "editor_used": editor_used,
                "editor_fallback": editor_fallback,
                "editor_attempts": editor_attempts,
                "legacy_used": legacy_used,
                "prompt_enhanced": bool(enhancer_meta.get("used")),
                "enhancer_model": enhancer_meta.get("model"),
            }
            result = {
                "content": content,
                "json_plan": json_plan,
                # Back-compat key (if any external uses it)
                "rendered_video": render_info,
                # Explicit fields expected by GUI
                "render_id": render_id,
                "render_status": render_status,
                "video_url": video_url,
                "analyses": analyzed_data,
                "agent_trace": agent_trace,
                "enhanced_prompt_preview": (enhanced_prompt or text_prompt)[:200] if isinstance(enhanced_prompt, str) else text_prompt[:200],
                "status": "success",
                "message": "Video editing plan created successfully!"
            }
            
            # Store editing plan
            if json_plan:
                plan_id = str(uuid.uuid4())
                editing_plan = EditingPlan(
                    content=content,
                    json_plan=json_plan,
                    media_files=media_files or [],
                    style="auto_detected",
                    duration=30.0,  # Default duration
                    timestamp=datetime.datetime.now().isoformat(),
                    plan_id=plan_id
                )
                self.memory.store_plan(editing_plan)
                result["plan_id"] = plan_id
            
            # Store session context if this is a new session
            if media_files and not hasattr(self, 'current_session_id'):
                self.current_session_id = self.memory.start_editing_session(media_files, text_prompt)
                if json_plan:
                    self.memory.sessions[self.current_session_id]["plans_generated"].append(plan_id)
                    self.memory._save_json(self.memory.sessions_file, self.memory.sessions)
            
            print("✅ Workflow completed successfully!")
            return result
        
        except Exception as e:
            print(f"❌ Workflow error: {str(e)}")
            return self._error_response(f"Processing failed: {str(e)}")

    def director_stage(
        self,
        text_prompt: str,
        media_files: List[str] = None,
        url_mappings: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Run up to Director Agent and return content + abstract_plan immediately.
        This is used by the SSE endpoint to stream the Director content with typing effect
        before continuing with Editor/Renderer.
        """
        try:
            # MEDIA ANALYSIS (reuse logic but minimal)
            analyzed_data: Dict[str, Any] = {}
            if media_files:
                files_to_analyze = []
                for file_path in media_files:
                    temp_filename = Path(file_path).name
                    cached = None
                    if url_mappings and temp_filename in url_mappings:
                        cloud_url = url_mappings[temp_filename]
                        cached = self.memory.get_analysis_by_url(cloud_url)
                    if not cached:
                        cached = self.memory.get_latest_analysis(temp_filename)
                    if cached:
                        analyzed_data[temp_filename] = {
                            "file_path": cached.file_path,
                            "file_type": cached.file_type,
                            "analysis": cached.analysis,
                            "metadata": cached.metadata,
                            "status": cached.status,
                            "cloud_url": cached.cloud_url,
                        }
                    else:
                        files_to_analyze.append(file_path)
                if files_to_analyze:
                    new_analyzed = self.media_analyser.analyze(files_to_analyze)
                    if url_mappings:
                        for filename, analysis in new_analyzed.items():
                            if isinstance(analysis, dict):
                                file_path = analysis.get("file_path", "")
                                temp_name = Path(file_path).name if file_path else filename
                                if temp_name in url_mappings:
                                    cloud_url = url_mappings[temp_name]
                                    analysis["cloud_url"] = cloud_url
                                    analysis["file_path"] = cloud_url
                    analyzed_data.update(new_analyzed)

            # PROMPT ENHANCEMENT
            enhanced_prompt, enhancer_meta = self.prompt_enhancer.enhance(text_prompt, analyzed_data)
            dp = enhanced_prompt if enhanced_prompt is not None else text_prompt

            # DIRECTOR AGENT
            content, abstract_plan = self.director_agent.plan(dp, analyzed_data)

            print(f"🔍 DEBUG: Director Agent returned content: {repr(content)}")
            print(f"🔍 DEBUG: Director Agent returned abstract_plan keys: {list(abstract_plan.keys()) if isinstance(abstract_plan, dict) else 'Not dict'}")

            # Return early payload
            return {
                "status": "ok",
                "content": content,
                "abstract_plan": abstract_plan,
                "analyses": analyzed_data,
                "enhanced_prompt_preview": (enhanced_prompt or text_prompt)[:200] if isinstance(enhanced_prompt, str) else text_prompt[:200],
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    def continue_after_director(
        self,
        content: str,
        abstract_plan: Dict[str, Any],
        analyzed_data: Dict[str, Any],
        text_prompt: str,
        media_files: List[str] = None,
        url_mappings: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Continue pipeline from Editor Agent through validation and optional rendering.
        Returns the same shape as process_request().
        """
        try:
            json_plan = None
            render_info = None
            render_id = None
            render_status = None
            video_url = None

            # MUSIC GENERATION (optional) prior to Editor
            try:
                if isinstance(abstract_plan, dict):
                    music_script = abstract_plan.get("music_script")
                    if isinstance(music_script, str) and music_script.strip():
                        print("🎵 Music script detected — generating soundtrack…")
                        print(f"   Music script preview: {music_script[:100]}...")
                        music_url = self.music_generator.generate(music_script)
                        if isinstance(music_url, str) and music_url.startswith("http"):
                            synth_name = "generated_music.mp3"
                            analyzed_data = dict(analyzed_data or {})
                            analyzed_data[synth_name] = {
                                "file_path": music_url,
                                "file_type": "audio",
                                "analysis": "AI-generated background music",
                                "metadata": {"source": "music_generator"},
                                "status": "success",
                                "cloud_url": music_url,
                            }
                            if url_mappings is None:
                                url_mappings = {}
                            url_mappings[synth_name] = music_url
                            print(f"✅ Music generated and injected: {music_url}")
                        else:
                            print(f"⚠️ Music generation returned no valid URL: {music_url}")
                    else:
                        print("ℹ️ No music_script in abstract_plan — skipping music generation")
                else:
                    print("ℹ️ abstract_plan is not a dict — skipping music generation")
            except Exception as me:
                print(f"⚠️ Music generation exception: {me}")
                import traceback
                traceback.print_exc()

            # EDITOR AGENT
            if abstract_plan is not None:
                json_plan = self.editor_agent.build_timeline(abstract_plan, analyzed_data, url_mappings)

            # VALIDATION + URL FIXING
            if json_plan:
                json_plan = self._validate_and_normalize_plan(json_plan)
                needs_url_fix = self._check_if_urls_need_fixing(json_plan)
                if needs_url_fix:
                    allowed_names = set(analyzed_data.keys()) if analyzed_data else set()
                    if media_files:
                        allowed_names.update({Path(p).name for p in media_files})
                    json_plan = self._enforce_allowed_assets(json_plan, allowed_names)
                    json_plan = self._map_temp_filenames_to_urls(json_plan, analyzed_data)
                    json_plan = self._force_replace_invalid_urls(json_plan, analyzed_data, url_mappings)
                    json_plan = self._validate_and_normalize_plan(json_plan)

            # RENDERER (optional based on existing config)
            if json_plan:
                if self.timeline_schema is not None:
                    try:
                        validate(instance=json_plan, schema=self.timeline_schema)
                    except ValidationError as ve:
                        render_info = {
                            "status": "validation_error",
                            "message": f"JSON plan failed schema validation: {ve.message}",
                            "render_id": None,
                            "video_url": None,
                        }
                        render_id = None
                        render_status = "validation_error"
                        video_url = None
                    else:
                        render_info = self.shotstack_renderer.render_video(json_plan)
                else:
                    render_info = self.shotstack_renderer.render_video(json_plan)

                if isinstance(render_info, dict):
                    render_id = render_info.get("render_id")
                    render_status = render_info.get("status")
                    video_url = render_info.get("video_url")

            result = {
                "content": content,
                "json_plan": json_plan,
                "rendered_video": render_info,
                "render_id": render_id,
                "render_status": render_status,
                "video_url": video_url,
                "analyses": analyzed_data,
                "status": "success",
                "message": "Video editing plan created successfully!",
            }
            return result
        except Exception as e:
            return self._error_response(f"Processing failed: {str(e)}")

    def _enforce_allowed_assets(self, plan: Dict[str, Any], allowed_filenames: set) -> Dict[str, Any]:
        """Ensure all asset srcs referenced in the plan are from the allowed set.
        If not allowed, replace src with 'placeholder' and add a note into clip if possible.
        """
        try:
            timeline = plan.get("timeline", {})
            tracks = timeline.get("tracks") or timeline.get("clips")
            if not tracks:
                return plan

            def sanitize_clip(clip: Dict[str, Any]):
                asset = clip.get("asset") or {}
                src = asset.get("src")
                if isinstance(src, str) and src:
                    fname = Path(src).name
                    if fname not in allowed_filenames:
                        if ASSET_ENFORCEMENT_MODE == "remove":
                            clip["__remove__"] = True
                        else:
                            asset["src"] = "placeholder"
                            # Optional note for transparency
                            clip["note"] = "Non-provided media reference replaced with placeholder"
                            clip["asset"] = asset

            # tracks can be list of track dicts with 'clips', or flat clips under 'clips'
            if isinstance(tracks, list):
                for t in tracks:
                    clips = t.get("clips", []) if isinstance(t, dict) else []
                    # sanitize in place
                    kept = []
                    for c in clips:
                        if isinstance(c, dict):
                            sanitize_clip(c)
                            if not c.get("__remove__"):
                                kept.append(c)
                    if isinstance(t, dict):
                        t["clips"] = kept
            elif isinstance(tracks, dict):
                # Uncommon shape
                kept = []
                for c in tracks.get("clips", []):
                    if isinstance(c, dict):
                        sanitize_clip(c)
                        if not c.get("__remove__"):
                            kept.append(c)
                tracks["clips"] = kept

            return plan
        except Exception:
            return plan

    def _map_temp_filenames_to_urls(self, plan: Dict[str, Any], analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Replace temp filenames in asset srcs with actual cloud URLs from analyses"""
        try:
            if not isinstance(plan, dict) or not analyses:
                return plan
            
            # Build filename -> cloud_url mapping from analyses
            filename_to_url = {}
            for fname, analysis in analyses.items():
                if isinstance(analysis, dict):
                    cloud_url = analysis.get("cloud_url")
                    file_path = analysis.get("file_path", "")
                    temp_name = Path(file_path).name if file_path else fname
                    if cloud_url:
                        filename_to_url[temp_name] = cloud_url
                        # Also map the base filename
                        filename_to_url[fname] = cloud_url
            
            if not filename_to_url:
                return plan
            
            print(f"🔗 Mapping {len(filename_to_url)} temp filenames to cloud URLs...")
            
            # Walk through plan and replace src fields
            timeline = plan.get("timeline", {})
            tracks = timeline.get("tracks", [])
            if isinstance(tracks, list):
                for track in tracks:
                    if not isinstance(track, dict):
                        continue
                    clips = track.get("clips", [])
                    if isinstance(clips, list):
                        for clip in clips:
                            if not isinstance(clip, dict):
                                continue
                            asset = clip.get("asset")
                            if isinstance(asset, dict):
                                src = asset.get("src")
                                if isinstance(src, str) and src:
                                    # Check if it's a temp filename
                                    if src in filename_to_url:
                                        old_src = src
                                        asset["src"] = filename_to_url[src]
                                        print(f"  ✅ Mapped '{old_src}' → '{filename_to_url[src][:60]}...'")
                                    # Also check basename
                                    elif Path(src).name in filename_to_url:
                                        old_src = src
                                        asset["src"] = filename_to_url[Path(src).name]
                                        print(f"  ✅ Mapped '{old_src}' → '{filename_to_url[Path(src).name][:60]}...'")
            
            return plan
        except Exception as e:
            print(f"⚠️ URL mapping error: {e}")
            return plan
    
    def _force_replace_invalid_urls(self, plan: Dict[str, Any], analyses: Dict[str, Any], url_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        AGGRESSIVE URL REPLACEMENT - Replace ANY invalid/example URLs with user's actual media.
        This ensures Shotstack NEVER receives dummy URLs like earth.mp4.
        """
        try:
            if not isinstance(plan, dict):
                return plan
            
            # Build list of valid user URLs from multiple sources
            user_urls = []
            
            # Source 1: From analyses (cloud_url field)
            if analyses:
                for fname, analysis in analyses.items():
                    if isinstance(analysis, dict):
                        cloud_url = analysis.get("cloud_url")
                        if cloud_url and cloud_url.startswith("https://res.cloudinary.com"):
                            user_urls.append(cloud_url)
            
            # Source 2: From url_mappings (passed from API)
            if url_mappings:
                for url in url_mappings.values():
                    if url and url.startswith("https://res.cloudinary.com"):
                        if url not in user_urls:
                            user_urls.append(url)
            
            if not user_urls:
                print("⚠️ No valid user URLs found - cannot replace invalid URLs")
                return plan
            
            print(f"🔧 Force-replacing invalid URLs with {len(user_urls)} user URL(s)...")
            
            # Define patterns of invalid/example URLs to replace
            invalid_patterns = [
                "shotstack-assets.s3",
                "shotstack.io/assets",
                "example.com",
                "placeholder",
                "earth.mp4",
                "drops.mp4",
                "motions.mp3"
            ]
            
            # Walk through plan and replace invalid URLs
            timeline = plan.get("timeline", {})
            tracks = timeline.get("tracks", [])
            replacement_count = 0
            
            if isinstance(tracks, list):
                for track_idx, track in enumerate(tracks):
                    if not isinstance(track, dict):
                        continue
                    clips = track.get("clips", [])
                    if isinstance(clips, list):
                        for clip_idx, clip in enumerate(clips):
                            if not isinstance(clip, dict):
                                continue
                            asset = clip.get("asset")
                            if isinstance(asset, dict):
                                src = asset.get("src")
                                asset_type = asset.get("type")
                                
                                # Only process video/audio/image assets
                                if asset_type not in ["video", "audio", "image"]:
                                    continue
                                
                                if isinstance(src, str) and src:
                                    is_invalid = False
                                    
                                    # Check if URL matches any invalid pattern
                                    for pattern in invalid_patterns:
                                        if pattern in src.lower():
                                            is_invalid = True
                                            break
                                    
                                    # Also check if it's NOT a Cloudinary URL
                                    if not src.startswith("https://res.cloudinary.com"):
                                        is_invalid = True
                                    
                                    if is_invalid:
                                        # Pick user URL (rotate through them if multiple clips)
                                        replacement_url = user_urls[clip_idx % len(user_urls)]
                                        old_src = src
                                        asset["src"] = replacement_url
                                        replacement_count += 1
                                        print(f"  🔄 Replaced invalid URL:")
                                        print(f"     ❌ '{old_src[:80]}...'")
                                        print(f"     ✅ '{replacement_url[:80]}...'")
            
            if replacement_count > 0:
                print(f"✅ Force-replaced {replacement_count} invalid URL(s) with user media")
            else:
                print("✅ All URLs are valid - no replacement needed")
            
            return plan
        except Exception as e:
            print(f"⚠️ Force URL replacement error: {e}")
            return plan
    
    def _check_if_urls_need_fixing(self, plan: Dict[str, Any]) -> bool:
        """
        Smart validation: Check if LLM already generated valid cloud URLs.
        Returns True if URLs need fixing, False if they're already valid.
        """
        try:
            if not isinstance(plan, dict):
                return False
            
            timeline = plan.get("timeline", {})
            tracks = timeline.get("tracks", [])
            
            if not isinstance(tracks, list):
                return False
            
            invalid_patterns = [
                "shotstack-assets.s3",
                "shotstack.io/assets",
                "example.com",
                "placeholder",
                "earth.mp4",
                "drops.mp4",
                "motions.mp3",
                "tmp"  # Catches tmpXXX.mp4
            ]
            
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                clips = track.get("clips", [])
                if isinstance(clips, list):
                    for clip in clips:
                        if not isinstance(clip, dict):
                            continue
                        asset = clip.get("asset")
                        if isinstance(asset, dict):
                            src = asset.get("src")
                            asset_type = asset.get("type")
                            
                            # Only check video/audio/image assets
                            if asset_type not in ["video", "audio", "image"]:
                                continue
                            
                            if isinstance(src, str) and src:
                                # Check if it's a valid Cloudinary URL
                                if not src.startswith("https://res.cloudinary.com"):
                                    return True  # Not a Cloudinary URL - needs fixing
                                
                                # Check for invalid patterns
                                for pattern in invalid_patterns:
                                    if pattern in src.lower():
                                        return True  # Has invalid pattern - needs fixing
            
            # All URLs are valid Cloudinary URLs
            return False
            
        except Exception as e:
            print(f"⚠️ URL validation check error: {e}")
            return True  # If error, play it safe and apply fixes
    
    def _validate_and_normalize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure basic schema presence and drop empty tracks/clips."""
        try:
            if not isinstance(plan, dict):
                return plan
            timeline = plan.setdefault("timeline", {})
            # Ensure tracks array
            tracks = timeline.get("tracks")
            if tracks is None:
                tracks = []
                timeline["tracks"] = tracks
            # Drop empty tracks
            if isinstance(tracks, list):
                cleaned_tracks = []
                for t in tracks:
                    if isinstance(t, dict):
                        clips = t.get("clips", [])
                        if isinstance(clips, list):
                            clips = [c for c in clips if isinstance(c, dict)]
                            t["clips"] = clips
                        if t.get("clips"):
                            cleaned_tracks.append(t)
                timeline["tracks"] = cleaned_tracks
            # Ensure output exists
            plan.setdefault("output", {"format": "mp4", "resolution": "1920x1080", "fps": 30})
            return plan
        except Exception:
            return plan

    def check_render_status(self, render_id: str) -> Dict[str, Any]:
        """Delegate to ShotstackRenderer to check render status and map fields."""
        try:
            status_info = self.shotstack_renderer.check_render_status(render_id)
            # Ensure consistent keys for GUI consumption
            return {
                "status": status_info.get("status", "unknown"),
                "progress": status_info.get("progress", 0),
                "video_url": status_info.get("video_url"),
                "message": status_info.get("message", "")
            }
        except Exception as e:
            return {
                "status": "error",
                "progress": 0,
                "video_url": None,
                "message": f"Status check error: {str(e)}"
            }
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response"""
        return {
            "content": message,
            "json_plan": None,
            "rendered_video": None,
            "analyses": {},
            "status": "error",
            "message": message
        }
    
    def refine_plan(self, plan_id: str, refinement_prompt: str) -> Dict[str, Any]:
        """Context-aware refinement using session history and media analysis"""
        try:
            print(f"🔄 Refining plan {plan_id} with context awareness...")
            
            # Get existing plan
            existing_plan = self.memory.get_plan(plan_id)
            if not existing_plan:
                return self._error_response("Plan not found")
            
            # Build comprehensive context using session history
            session_id = getattr(self, 'current_session_id', None)
            if session_id:
                # Use context-aware refinement with full session history
                context_prompt = self.memory.build_refinement_context(session_id, refinement_prompt)
                print("📋 Using session context for intelligent refinement")
            else:
                # Fallback to basic refinement
                # DON'T show the old JSON (might have wrong structure), just the description
                context_prompt = f"""
REFINE EXISTING VIDEO PLAN:
Original concept: {existing_plan.get('content', '')}
Refinement requested: {refinement_prompt}

Create an UPDATED Shotstack-compatible JSON timeline that incorporates the refinement.
Follow all the Shotstack schema rules provided above.
Ensure all properties are in the correct locations (fit, effect, filter, etc. go on CLIP, not in asset).
"""
                print("⚠️ No session context available, using basic refinement")
            
            # Process refinement with context
            llm_response = self.llm_processor.process({}, context_prompt)
            
            if llm_response.startswith("Error:"):
                return self._error_response(llm_response)
            
            # Extract refined plan
            content, json_plan = self.json_extractor.extract_and_separate(llm_response)

            # Enforce and normalize
            try:
                allowed_names = set()
                if session_id:
                    allowed_names.update({Path(p).name for p in self.memory.sessions[session_id].get("media_files", [])})
                # Also add media from existing plan
                allowed_names.update({Path(p).name for p in existing_plan.get('media_files', [])})
                if json_plan:
                    # Validate structure first
                    json_plan = self._validate_and_normalize_plan(json_plan)
                    
                    # Check if URLs need fixing
                    needs_url_fix = self._check_if_urls_need_fixing(json_plan)
                    
                    if needs_url_fix:
                        print("⚠️ Detected invalid URLs in refined plan - applying fixes...")
                        json_plan = self._enforce_allowed_assets(json_plan, allowed_names)
                        
                        # Get FRESH analyses from memory
                        session_analyses = {}
                        for media_file in existing_plan.get('media_files', []):
                            temp_filename = Path(media_file).name
                            cached_analysis = self.memory.get_latest_analysis(temp_filename)
                            if cached_analysis:
                                session_analyses[temp_filename] = {
                                    "file_path": cached_analysis.file_path,
                                    "file_type": cached_analysis.file_type,
                                    "cloud_url": cached_analysis.cloud_url,
                                    "analysis": cached_analysis.analysis,
                                    "metadata": cached_analysis.metadata,
                                    "status": cached_analysis.status
                                }
                        
                        json_plan = self._map_temp_filenames_to_urls(json_plan, session_analyses)
                        json_plan = self._force_replace_invalid_urls(json_plan, session_analyses, url_mappings=None)
                        json_plan = self._validate_and_normalize_plan(json_plan)
                    else:
                        print("✅ Refined plan has valid cloud URLs - no URL fixing needed!")
            except Exception as e:
                print(f"⚠️ Refinement asset enforcement skipped: {e}")
            
            # RENDER THE REFINED PLAN
            render_info = None
            render_id = None
            render_status = None
            video_url = None
            if json_plan:
                # Validate plan before attempting to render
                if self.timeline_schema is not None:
                    try:
                        validate(instance=json_plan, schema=self.timeline_schema)
                    except ValidationError as ve:
                        render_info = {
                            "status": "validation_error",
                            "message": f"JSON plan failed schema validation: {ve.message}",
                            "render_id": None,
                            "video_url": None
                        }
                        render_id = None
                        render_status = "validation_error"
                        video_url = None
                    else:
                        print("🎬 Step 5: Rendering refined plan via Shotstack...")
                        render_info = self.shotstack_renderer.render_video(json_plan)
                else:
                    # No schema available, proceed with best effort
                    print("🎬 Step 5: Rendering refined plan via Shotstack...")
                    render_info = self.shotstack_renderer.render_video(json_plan)
                # Map to expected fields
                if isinstance(render_info, dict):
                    render_id = render_info.get("render_id")
                    render_status = render_info.get("status")
                    video_url = render_info.get("video_url")
            
            # Create new plan for the refinement
            if json_plan:
                new_plan_id = str(uuid.uuid4())
                refined_plan = EditingPlan(
                    content=content,
                    json_plan=json_plan,
                    media_files=existing_plan.get('media_files', []),
                    style=existing_plan.get('style', 'refined'),
                    duration=existing_plan.get('duration', 30.0),
                    timestamp=datetime.datetime.now().isoformat(),
                    plan_id=new_plan_id
                )
                
                # Store the refined plan
                self.memory.store_plan(refined_plan)
                
                # Track refinement in session if available
                if session_id:
                    self.memory.add_refinement_to_session(
                        session_id, 
                        refinement_prompt, 
                        existing_plan, 
                        asdict(refined_plan)
                    )
                    # Update session with new plan
                    self.memory.sessions[session_id]["plans_generated"].append(new_plan_id)
                    self.memory.sessions[session_id]["current_plan"] = new_plan_id
                    self.memory._save_json(self.memory.sessions_file, self.memory.sessions)
                
                print("✅ Context-aware refinement completed successfully!")
                
                return {
                    "content": content,
                    "json_plan": json_plan,
                    "rendered_video": render_info,
                    "render_id": render_id,
                    "render_status": render_status,
                    "video_url": video_url,
                    "plan_id": new_plan_id,
                    "original_plan_id": plan_id,
                    "status": "success",
                    "message": "Plan refined with context awareness!"
                }
            else:
                return self._error_response("Failed to generate refined plan")
            
        except Exception as e:
            print(f"❌ Refinement error: {str(e)}")
            return self._error_response(f"Refinement failed: {str(e)}")
    
    def refine_with_new_media(self, plan_id: str, refinement_prompt: str, new_media_files: List[str], url_mappings: Dict[str, str] = None) -> Dict[str, Any]:
        """Enhanced refinement that analyzes new media and combines with existing analysis"""
        try:
            print(f"🔄 Enhanced refinement with {len(new_media_files)} new media files...")
            
            # Get existing plan
            existing_plan = self.memory.get_plan(plan_id)
            if not existing_plan:
                return self._error_response("Plan not found")
            
            # Get session context
            session_id = getattr(self, 'current_session_id', None)
            if not session_id:
                # Create a lightweight session context from the existing plan so refinement still works
                print("⚠️ No session context available, initializing a temporary context from existing plan")
                temp_media = existing_plan.get("media_files", []) or []
                try:
                    self.current_session_id = self.memory.start_editing_session(temp_media, refinement_prompt)
                    session_id = self.current_session_id
                except Exception:
                    # Fallback to a dummy session id-like behavior without persisting
                    session_id = "temp_refine_session"
            
            # STEP 1: Analyze new media files
            print("📹 Step 1: Analyzing new media files...")
            new_analyzed_data = self.media_analyser.analyze(new_media_files)
            
            # INJECT CLOUD URLs if provided (before LLM sees the data)
            if url_mappings:
                print("🔗 Injecting cloud URLs into new analyzed data...")
                for filename, analysis in new_analyzed_data.items():
                    if isinstance(analysis, dict):
                        file_path = analysis.get("file_path", "")
                        temp_name = Path(file_path).name if file_path else filename
                        if temp_name in url_mappings:
                            cloud_url = url_mappings[temp_name]
                            analysis["cloud_url"] = cloud_url
                            # SIMPLE FIX: Store cloud URL as file_path so LLM sees it directly!
                            analysis["file_path"] = cloud_url
                            print(f"✅ Set file_path to cloud_url for {filename}: {cloud_url[:60]}...")
            
            # Store new analyses (including cloud_url if present)
            for filename, analysis in new_analyzed_data.items():
                if isinstance(analysis, dict):
                    # Use cloud_url as file_path if available
                    cloud_url = analysis.get("cloud_url", "")
                    stored_file_path = cloud_url if cloud_url else analysis.get("file_path", "")
                    
                    media_analysis = MediaAnalysis(
                        file_path=stored_file_path,  # Cloudinary URL if available!
                        file_type=analysis.get("file_type", "unknown"),
                        filename=filename,
                        analysis=analysis.get("analysis", ""),
                        metadata=analysis.get("metadata", {}),
                        timestamp=datetime.datetime.now().isoformat(),
                        status=analysis.get("status", "success"),
                        cloud_url=cloud_url
                    )
                    self.memory.store_analysis(media_analysis)
                    print(f"💾 Cached new media: {filename} → {stored_file_path[:60] if stored_file_path else 'No path'}...")
            
            # STEP 2: Update session with new media files (normalized)
            normalized_new = [self.memory._normalize_path(f) for f in new_media_files]
            self.memory.sessions[session_id]["media_files"].extend(normalized_new)
            self.memory._save_json(self.memory.sessions_file, self.memory.sessions)
            
            # STEP 3: Build comprehensive context with ALL media (old + new)
            print("🧠 Step 2: Building comprehensive context with all media...")
            enhanced_context = self._build_enhanced_refinement_context(
                session_id, refinement_prompt, new_analyzed_data, existing_plan
            )
            
            # STEP 4: Process enhanced refinement
            print("🤖 Step 3: Processing enhanced refinement...")
            llm_response = self.llm_processor.process({}, enhanced_context)
            
            if llm_response.startswith("Error:"):
                return self._error_response(llm_response)
            
            # STEP 5: Extract refined plan
            print("📄 Step 4: Extracting refined plan...")
            content, json_plan = self.json_extractor.extract_and_separate(llm_response)

            # Enforce and normalize
            try:
                # Build allowed filenames from session + existing plan + new files
                allowed_names = set()
                try:
                    allowed_names.update({Path(p).name for p in self.memory.sessions.get(session_id, {}).get("media_files", [])})
                except Exception:
                    pass
                try:
                    allowed_names.update({Path(p).name for p in (existing_plan.get("media_files") or [])})
                except Exception:
                    pass
                try:
                    allowed_names.update({Path(p).name for p in (new_media_files or [])})
                except Exception:
                    pass
                if json_plan:
                    # Validate structure first
                    json_plan = self._validate_and_normalize_plan(json_plan)
                    
                    # Check if URLs need fixing
                    needs_url_fix = self._check_if_urls_need_fixing(json_plan)
                    
                    if needs_url_fix:
                        print("⚠️ Detected invalid URLs in enhanced refinement - applying fixes...")
                        json_plan = self._enforce_allowed_assets(json_plan, allowed_names)
                        json_plan = self._map_temp_filenames_to_urls(json_plan, new_analyzed_data)
                        json_plan = self._force_replace_invalid_urls(json_plan, new_analyzed_data, url_mappings)
                        json_plan = self._validate_and_normalize_plan(json_plan)
                    else:
                        print("✅ Enhanced refinement has valid cloud URLs - no URL fixing needed!")
            except Exception as e:
                print(f"⚠️ Enhanced refinement asset enforcement skipped: {e}")
            
            # STEP 6: Optionally render via Shotstack (same behavior as process_request)
            render_info = None
            render_id = None
            render_status = None
            video_url = None
            if json_plan:
                if self.timeline_schema is not None:
                    try:
                        validate(instance=json_plan, schema=self.timeline_schema)
                    except ValidationError as ve:
                        render_info = {
                            "status": "validation_error",
                            "message": f"JSON plan failed schema validation: {ve.message}",
                            "render_id": None,
                            "video_url": None
                        }
                        render_id = None
                        render_status = "validation_error"
                        video_url = None
                    else:
                        print("🎬 Rendering refined plan via Shotstack...")
                        render_info = self.shotstack_renderer.render_video(json_plan)
                else:
                    print("🎬 Rendering refined plan via Shotstack (no schema)...")
                    render_info = self.shotstack_renderer.render_video(json_plan)
                if isinstance(render_info, dict):
                    render_id = render_info.get("render_id")
                    render_status = render_info.get("status")
                    video_url = render_info.get("video_url")

            # STEP 7: Create new enhanced plan
            if json_plan:
                new_plan_id = str(uuid.uuid4())
                refined_plan = EditingPlan(
                    content=content,
                    json_plan=json_plan,
                    media_files=self.memory.sessions[session_id]["media_files"],  # All media files
                    style=existing_plan.get('style', 'enhanced'),
                    duration=existing_plan.get('duration', 30.0),
                    timestamp=datetime.datetime.now().isoformat(),
                    plan_id=new_plan_id
                )
                
                # Store the enhanced plan
                self.memory.store_plan(refined_plan)
                
                # Track refinement with new media in session
                refinement_note = f"{refinement_prompt} (+ {len(new_media_files)} new media files)"
                self.memory.add_refinement_to_session(
                    session_id, 
                    refinement_note, 
                    existing_plan, 
                    asdict(refined_plan)
                )
                
                # Update session with new plan
                self.memory.sessions[session_id]["plans_generated"].append(new_plan_id)
                self.memory.sessions[session_id]["current_plan"] = new_plan_id
                self.memory._save_json(self.memory.sessions_file, self.memory.sessions)
                
                print("✅ Enhanced refinement with new media completed successfully!")
                
                return {
                    "content": content,
                    "json_plan": json_plan,
                    "rendered_video": render_info,
                    "render_id": render_id,
                    "render_status": render_status,
                    "video_url": video_url,
                    "plan_id": new_plan_id,
                    "original_plan_id": plan_id,
                    "analyses": new_analyzed_data,  # Include new analyses in response
                    "status": "success",
                    "message": f"Plan enhanced with {len(new_media_files)} new media files!"
                }
            else:
                return self._error_response("Failed to generate enhanced plan")
                
        except Exception as e:
            print(f"❌ Enhanced refinement error: {str(e)}")
            return self._error_response(f"Enhanced refinement failed: {str(e)}")
    
    def _build_enhanced_refinement_context(self, session_id: str, refinement_prompt: str, 
                                         new_analyzed_data: Dict[str, Any], existing_plan: Dict[str, Any]) -> str:
        """Build enhanced context that includes new media analysis with existing context"""
        
        # Get existing session context
        base_context = self.memory.build_refinement_context(session_id, refinement_prompt)
        
        # Add new media analysis section
        new_media_section = "\n\nNEW MEDIA ANALYSIS:\n" + "="*20 + "\n"
        
        for filename, analysis in new_analyzed_data.items():
            new_media_section += f"📄 {filename} (NEW):\n"
            new_media_section += f"   Type: {analysis.get('file_type', 'unknown')}\n"
            new_media_section += f"   Analysis: {analysis.get('analysis', 'N/A')[:300]}...\n\n"
        
        # Enhanced instructions
        enhanced_instructions = """
ENHANCED REFINEMENT INSTRUCTIONS:
================================
You now have access to BOTH the original media analysis AND new media files.

Please create a refined editing plan that:
1. INTEGRATES the new media with existing media seamlessly
2. MAINTAINS consistency with the original media's style and mood
3. INCORPORATES the new refinement requirements
4. USES the new media to ENHANCE the overall video (not replace existing elements)
5. ENSURES proper timing and synchronization across all media
6. CREATES a cohesive narrative that leverages all available assets

The new media should COMPLEMENT and ENHANCE the existing plan, not override it.
"""
        
        return base_context + new_media_section + enhanced_instructions
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return self.current_session.copy()
    
    def clear_session(self):
        """Clear current session data"""
        self.current_session = {
            "media_files": [],
            "analyses": {},
            "current_plan": None,
            "render_history": [],
            "conversation_context": []
        }
        print("🧹 Session cleared")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Get statistics from all components
        llm_stats = self.llm_processor.get_processor_stats()
        json_stats = self.json_extractor.get_extraction_stats()
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "llm_processor": llm_stats,
            "json_extractor": json_stats,
            "error_handler": error_stats,
            "session_info": self.current_session
        }
    
    def generate_system_report(self) -> str:
        """Generate a comprehensive system report"""
        
        stats = self.get_system_statistics()
        
        report = f"""
🎬 **Advanced Video Editing Agent - System Report**

**LLM Processor Statistics:**
• Total Requests: {stats['llm_processor']['total_requests']}
• Successful Requests: {stats['llm_processor']['successful_requests']}
• Success Rate: {stats['llm_processor']['success_rate']}

**JSON Extractor Statistics:**
• Total Attempts: {stats['json_extractor']['total_attempts']}
• Successful Extractions: {stats['json_extractor']['successful_extractions']}
• Success Rate: {stats['json_extractor']['success_rate']}

**Error Handler Statistics:**
• Total Errors: {stats['error_handler']['total_errors']}
• Successful Recoveries: {stats['error_handler']['total_recoveries']}
• Recovery Rate: {stats['error_handler']['recovery_rate']}

**Current Session:**
• Media Files: {len(stats['session_info']['media_files'])}
• Analyses: {len(stats['session_info']['analyses'])}
• Render History: {len(stats['session_info']['render_history'])}
"""
        
        return report


print("✅ Clean Video Editing Agent loaded successfully!")
