#!/usr/bin/env python3
"""
Google Gemini API Diagnostic Script
Run this to identify the exact issue with Gemini API
"""

import os
import requests
import json
from dotenv import load_dotenv

def diagnose_gemini_api():
    """Comprehensive Gemini API diagnostics"""
    print("🔍 Google Gemini API Diagnostics Starting...\n")
    
    # Step 1: Check .env file
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    if not gemini_key:
        print("❌ ISSUE FOUND: No GEMINI_API_KEY in .env file")
        print("🔧 FIX: Add GEMINI_API_KEY=your_key_here to .env file")
        print("📝 Get key from: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✅ Gemini key found: {gemini_key[:8]}...{gemini_key[-4:]}")
    
    # Step 2: Test API key validity
    print("\n🧪 Testing API key validity...")
    
    try:
        # Test with models endpoint first
        models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
        
        response = requests.get(models_url, timeout=10)
        
        print(f"Models endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            models_data = response.json()
            if "models" in models_data:
                model_count = len(models_data["models"])
                print(f"✅ API key is valid - found {model_count} available models")
            else:
                print("⚠️  Valid response but no models found")
        elif response.status_code == 400:
            print("❌ ISSUE: Invalid API key or malformed request")
            print("🔧 FIX: Check your API key at https://makersuite.google.com/app/apikey")
            return False
        elif response.status_code == 403:
            print("❌ ISSUE: API key lacks permissions or quota exceeded")
            print("🔧 FIX: Check API quotas in Google AI Studio")
            return False
        elif response.status_code == 404:
            print("❌ ISSUE: Incorrect API endpoint")
            return False
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Step 3: Test content generation
    print("\n🤖 Testing content generation...")
    
    success = test_gemini_generation(gemini_key)
    
    if success:
        print("✅ Gemini API is working correctly!")
        return True
    else:
        print("❌ Content generation failed")
        return False

def test_gemini_generation(api_key: str) -> bool:
    """Test actual content generation"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
        # Simple test payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Hello, how are you?"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 50
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        print("  Making generation request...")
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        print(f"  Generation status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"  ✅ Generated text: {generated_text[:100]}...")
                return True
            else:
                print("  ❌ No content in response")
                print(f"  Response: {json.dumps(result, indent=2)}")
        else:
            print(f"  ❌ Generation failed: {response.status_code}")
            print(f"  Error: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"  ❌ Generation exception: {e}")
        return False

def check_api_quotas():
    """Provide guidance on checking quotas"""
    print("\n📊 API Quota Information:")
    print("• Free tier: 60 requests per minute")
    print("• Daily limit varies by region")
    print("• Check usage at: https://makersuite.google.com/app/apikey")
    print("• If exceeded, wait for quota reset")

def show_setup_instructions():
    """Show correct setup steps"""
    print("\n📝 Correct Gemini Setup:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    print("5. Add to .env file: GEMINI_API_KEY=your_key_here")
    print("6. Restart your application")

if __name__ == "__main__":
    print("=" * 60)
    print("    Google Gemini API Connection Diagnostic")
    print("=" * 60)
    
    # Run diagnostics
    success = diagnose_gemini_api()
    
    if not success:
        check_api_quotas()
        show_setup_instructions()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 GEMINI API WORKING!")
        print("Your resume analyzer should now provide AI feedback.")
    else:
        print("🔧 GEMINI API ISSUES DETECTED")
        print("\nMOST COMMON FIXES:")
        print("1. Get valid API key from Google AI Studio")
        print("2. Check if you've exceeded free tier quotas")
        print("3. Ensure API key has proper permissions")
        print("4. Wait a few minutes and try again")
    
    print("=" * 60)