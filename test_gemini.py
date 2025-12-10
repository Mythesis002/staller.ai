#!/usr/bin/env python
"""
Test script to verify Gemini API configuration and connectivity
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test Gemini API with different models"""
    
    print("=" * 80)
    print("🧪 GEMINI API TEST SUITE")
    print("=" * 80)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("💡 Add it to your .env file: GOOGLE_API_KEY=your_key_here")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Try to import and initialize Gemini
    try:
        from google import genai
        print("✅ google-genai package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import google-genai: {e}")
        print("💡 Install it with: pip install google-genai")
        return
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini client initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        return
    
    # Test different models
    models_to_test = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash-exp",
    ]
    
    print("🔍 Testing different Gemini models...")
    print("-" * 80)
    
    for model_name in models_to_test:
        print(f"\n📡 Testing model: {model_name}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Say 'Hello! I am working.' in exactly 5 words."
            )
            
            result = response.text if hasattr(response, 'text') else str(response)
            print(f"✅ SUCCESS!")
            print(f"   Response: {result.strip()[:100]}")
            
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg:
                print(f"⚠️  Model overloaded (503): Try again later")
            elif "404" in error_msg:
                print(f"❌ Model not found (404): Invalid model name")
            elif "429" in error_msg:
                print(f"⚠️  Rate limited (429): Too many requests")
            elif "401" in error_msg or "403" in error_msg:
                print(f"❌ Authentication failed: Check your API key")
            else:
                print(f"❌ Error: {error_msg[:150]}")
    
    print()
    print("=" * 80)
    print("🎯 RECOMMENDED MODEL FOR YOUR PROJECT")
    print("=" * 80)
    print("Based on testing, use this in your config.py or .env:")
    print("GEMINI_MODEL=gemini-1.5-flash-latest")
    print()
    print("Or for better quality (slower, more expensive):")
    print("GEMINI_MODEL=gemini-1.5-pro-latest")
    print("=" * 80)


def test_video_editor_config():
    """Test the actual configuration used by video editor"""
    print("\n" + "=" * 80)
    print("🎬 VIDEO EDITOR CONFIGURATION TEST")
    print("=" * 80)
    
    try:
        from config import GOOGLE_API_KEY, GEMINI_MODEL, ENABLE_PROMPT_ENHANCER
        
        print(f"API Key: {'✅ Set' if GOOGLE_API_KEY else '❌ Not set'}")
        print(f"Model: {GEMINI_MODEL}")
        print(f"Prompt Enhancer: {'✅ Enabled' if ENABLE_PROMPT_ENHANCER else '❌ Disabled'}")
        
        if GOOGLE_API_KEY:
            print("\n🧪 Testing with video editor configuration...")
            from google import genai
            client = genai.Client(api_key=GOOGLE_API_KEY)
            
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Respond with: 'Video editor config works!'"
            )
            
            result = response.text if hasattr(response, 'text') else str(response)
            print(f"✅ SUCCESS with {GEMINI_MODEL}")
            print(f"   Response: {result.strip()}")
        
    except Exception as e:
        print(f"❌ Error testing video editor config: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    test_gemini_api()
    test_video_editor_config()
    
    print("\n✨ Test complete! Check results above.")
    print("💡 If all tests pass, restart your video editor application.")
