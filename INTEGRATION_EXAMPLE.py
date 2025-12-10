#!/usr/bin/env python
"""
Example: Integrating TransitionAgent into the video editing workflow

Workflow: DirectorAgent → TransitionAgent → EditorAgent → Shotstack
"""

from video_editing_agent import (
    MediaAnalyzer,
    DirectorAgent,
    TransitionAgent,  # NEW
    EditorAgent,
    ShotstackRenderer
)

def create_video_with_transitions(
    user_prompt: str,
    media_files: dict,
    url_mappings: dict = None
):
    """
    Complete workflow with transition selection
    
    Args:
        user_prompt: User's creative request
        media_files: Dict of {filename: file_path}
        url_mappings: Dict of {filename: cloudinary_url}
    """
    
    print("🎬 Starting Video Creation with Smart Transitions\n")
    
    # ========================================
    # STEP 1: Analyze Media
    # ========================================
    print("📊 STEP 1: Analyzing Media...")
    analyzer = MediaAnalyzer()
    analyzed_data = {}
    
    for filename, filepath in media_files.items():
        if filepath.endswith(('.mp4', '.mov', '.avi')):
            analysis = analyzer.analyze_video(filepath)
        elif filepath.endswith(('.mp3', '.wav', '.m4a')):
            analysis = analyzer.analyze_audio(filepath)
        elif filepath.endswith(('.jpg', '.png', '.jpeg')):
            analysis = analyzer.analyze_image(filepath)
        else:
            continue
        
        # Inject cloud URL if provided
        if url_mappings and filename in url_mappings:
            analysis.cloud_url = url_mappings[filename]
        
        analyzed_data[filename] = {
            "file_path": analysis.file_path,
            "file_type": analysis.file_type,
            "analysis": analysis.analysis,
            "cloud_url": analysis.cloud_url,
            "metadata": analysis.metadata
        }
    
    print(f"✅ Analyzed {len(analyzed_data)} media files\n")
    
    # ========================================
    # STEP 2: Director Creates Plan
    # ========================================
    print("🎭 STEP 2: Director Creating Creative Plan...")
    director = DirectorAgent()
    director_plan = director.create_plan(user_prompt, analyzed_data)
    
    print(f"✅ Director plan created:")
    print(f"   Style: {director_plan.get('style', 'N/A')}")
    print(f"   Mood: {director_plan.get('mood', 'N/A')}")
    print(f"   Pacing: {director_plan.get('pacing', 'N/A')}")
    print(f"   Duration: {director_plan.get('target_duration', 'N/A')}s\n")
    
    # ========================================
    # STEP 3: Transition Agent Selects Transitions ← NEW
    # ========================================
    print("🎨 STEP 3: Transition Agent Selecting Transitions...")
    transition_agent = TransitionAgent(library_path="transition_library.json")
    enriched_plan = transition_agent.select_transitions(director_plan, analyzed_data)
    
    transitions = enriched_plan.get('transitions', [])
    print(f"✅ Selected {len(transitions)} transitions:")
    for i, trans_data in enumerate(transitions):
        transition = trans_data.get('transition', {})
        print(f"   {i+1}. {transition.get('name', 'Unknown')} ({transition.get('duration', 0)}s)")
    print()
    
    # ========================================
    # STEP 4: Editor Builds Timeline
    # ========================================
    print("🎬 STEP 4: Editor Building Shotstack Timeline...")
    editor = EditorAgent()
    timeline = editor.build_timeline(enriched_plan, analyzed_data, url_mappings)
    
    print(f"✅ Timeline created with {len(timeline.get('timeline', {}).get('tracks', []))} tracks\n")
    
    # ========================================
    # STEP 5: Render with Shotstack
    # ========================================
    print("🎥 STEP 5: Rendering with Shotstack...")
    renderer = ShotstackRenderer()
    render_result = renderer.render_video(timeline)
    
    print(f"✅ Render submitted:")
    print(f"   Render ID: {render_result.get('render_id', 'N/A')}")
    print(f"   Status: {render_result.get('status', 'N/A')}\n")
    
    return {
        "analyzed_data": analyzed_data,
        "director_plan": director_plan,
        "enriched_plan": enriched_plan,
        "timeline": timeline,
        "render_result": render_result
    }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    
    # Example 1: Simple video with transitions
    media_files = {
        "intro.mp4": "/path/to/intro.mp4",
        "main.mp4": "/path/to/main.mp4",
        "outro.mp4": "/path/to/outro.mp4"
    }
    
    url_mappings = {
        "intro.mp4": "https://res.cloudinary.com/demo/video/upload/v1/intro.mp4",
        "main.mp4": "https://res.cloudinary.com/demo/video/upload/v1/main.mp4",
        "outro.mp4": "https://res.cloudinary.com/demo/video/upload/v1/outro.mp4"
    }
    
    user_prompt = "Create an exciting viral video with fast cuts and dynamic energy"
    
    result = create_video_with_transitions(
        user_prompt=user_prompt,
        media_files=media_files,
        url_mappings=url_mappings
    )
    
    print("=" * 80)
    print("🎉 VIDEO CREATION COMPLETE!")
    print("=" * 80)
    print(f"Final Timeline Tracks: {len(result['timeline'].get('timeline', {}).get('tracks', []))}")
    print(f"Transitions Used: {len(result['enriched_plan'].get('transitions', []))}")
    print(f"Render ID: {result['render_result'].get('render_id', 'N/A')}")


# ============================================================
# DIRECT TRANSITION AGENT USAGE
# ============================================================

def test_transition_selection():
    """Test TransitionAgent selection independently"""
    
    # Mock director plan
    director_plan = {
        "style": "viral",
        "pacing": "fast",
        "mood": "energetic",
        "target_duration": 15,
        "tracks": [
            {
                "role": "main_video",
                "media_files": ["clip1.mp4", "clip2.mp4", "clip3.mp4"],
                "timing_notes": "Fast cuts every 5 seconds"
            }
        ],
        "key_moments": []
    }
    
    # Initialize agent
    transition_agent = TransitionAgent()
    
    # Select transitions
    enriched_plan = transition_agent.select_transitions(director_plan, {})
    
    # Print results
    print("\n🎨 Transition Selection Results:")
    print("=" * 60)
    
    for trans_data in enriched_plan.get('transitions', []):
        transition = trans_data['transition']
        cut = trans_data['cut_point']
        
        print(f"\nTransition: {transition['name']}")
        print(f"  Duration: {transition['duration']}s")
        print(f"  URL: {transition['url']}")
        print(f"  ChromaKey: {transition['chromaKey']}")
        print(f"  Categories: {', '.join(transition['categories'])}")
        print(f"  Moods: {', '.join(transition['moods'])}")
        print(f"  Cut: {cut.get('from_file', '?')} → {cut.get('to_file', '?')}")


if __name__ == "__main__":
    # Uncomment to test transition selection only
    # test_transition_selection()
    pass
