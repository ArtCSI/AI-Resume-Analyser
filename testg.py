import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY", "").strip()

print("=== GEMINI API QUICK TEST ===")
print(f"API Key found: {'YES' if api_key else 'NO'}")

if api_key:
    print(f"Key preview: {api_key[:10]}...{api_key[-5:]}")
    
    # Test the API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": "Say hello"}]}]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ GEMINI API WORKING!")
        elif response.status_code == 400:
            print("❌ Bad Request - Check API key")
        elif response.status_code == 403:
            print("❌ Forbidden - API key invalid or quota exceeded")
        else:
            print("❌ Other error")
            
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No API key found in .env file")
    print("Add: GEMINI_API_KEY=your_key_here")