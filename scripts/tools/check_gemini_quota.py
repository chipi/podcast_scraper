#!/usr/bin/env python3
"""Check Gemini API quota and rate limits.

This script helps diagnose 429 rate limit errors by:
1. Testing a simple API call to see if it works
2. Checking what error details are available
3. Providing guidance on rate limits
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import google.genai as genai
except ImportError:
    print("ERROR: google-genai package not installed")
    print("Install with: pip install 'podcast-scraper[gemini]'")
    sys.exit(1)

from dotenv import load_dotenv

# Load .env file
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment")
    print("Set it in .env file or export GEMINI_API_KEY=your_key")
    sys.exit(1)

print("=" * 80)
print("Gemini API Quota Checker")
print("=" * 80)
# Security: Don't log API key, even partially masked
if api_key:
    print("API Key: [REDACTED] (configured)")
else:
    print("API Key: [NOT SET]")
print()

# Configure Gemini
genai.configure(api_key=api_key)

# Test 1: Simple text generation
print("Test 1: Simple text generation (gemini-2.0-flash)")
print("-" * 80)
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say 'Hello' in one word.")
    print(f"✅ SUCCESS: {response.text}")
    print()
except Exception as e:
    error_type = type(e).__name__
    error_msg = str(e)
    print(f"❌ FAILED: {error_type}: {error_msg}")

    # Check for 429/rate limit
    if (
        "429" in error_msg
        or "quota" in error_msg.lower()
        or "rate limit" in error_msg.lower()
        or "resource exhausted" in error_msg.lower()
    ):
        print("\n⚠️  RATE LIMIT ERROR DETECTED")
        print("\nPossible causes:")
        print("1. Too many requests per minute - you may have hit the RPM limit")
        print(
            "2. Daily quota exceeded - check your quota at https://aistudio.google.com/app/apikey"
        )
        print("3. Free tier limits - free tier has lower limits than paid")
        print("\nError details:")
        print(f"  Type: {error_type}")
        print(f"  Message: {error_msg}")
        if hasattr(e, "status_code"):
            print(f"  Status code: {e.status_code}")
        if hasattr(e, "response"):
            print(f"  Response: {e.response}")
        if hasattr(e, "retry_after"):
            print(f"  Retry after: {e.retry_after} seconds")
    else:
        print(f"\nFull error: {e}")
        if hasattr(e, "__dict__"):
            print(f"Error attributes: {e.__dict__}")
    print()

# Test 2: Check model availability
print("Test 2: List available models")
print("-" * 80)
try:
    models = list(genai.list_models())
    print(f"✅ Found {len(models)} models")
    # Filter for generation models
    generation_models = [m for m in models if "generateContent" in m.supported_generation_methods]
    print(f"   {len(generation_models)} support generateContent")
    if generation_models:
        print("   Examples:")
        for model in generation_models[:5]:
            print(f"     - {model.name}")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    print()

# Summary
print("=" * 80)
print("Summary")
print("=" * 80)
print("If you're getting 429 errors:")
print("1. Check your quota: https://aistudio.google.com/app/apikey")
print("2. Wait a few minutes between test runs")
print("3. Reduce parallelism (process episodes sequentially)")
print("4. Check if you're on free tier (has lower limits)")
print()
print("Gemini API rate limits (approximate):")
print("- Free tier: ~15 requests per minute")
print("- Paid tier: Higher limits (varies by account)")
print("- Daily quotas also apply")
print()
print("For more info: https://ai.google.dev/pricing")
