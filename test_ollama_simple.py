#!/usr/bin/env python3
"""
Simple Ollama DeepSeek Test
"""

import requests
import json

def test_ollama_connection():
    """Test basic connection to Ollama DeepSeek"""
    
    print("🤖 Testing Ollama DeepSeek Connection")
    print("=" * 50)
    
    api_url = "http://localhost:11434/api/generate"
    
    # Simple test prompt
    payload = {
        "model": "deepseek-coder:latest",
        "prompt": "Hello! Can you help me analyze a software development problem? Please respond with just 'Yes, I can help with software analysis.'",
        "stream": False,
        "options": {
            "num_predict": 50,
            "temperature": 0.1
        }
    }
    
    try:
        print("📡 Sending request to Ollama...")
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        ai_response = result.get("response", "").strip()
        
        print(f"✅ DeepSeek Response: {ai_response}")
        print(f"📊 Response length: {len(ai_response)} characters")
        print("🎉 Connection successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_ollama_connection()