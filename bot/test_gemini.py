import os
import google.generativeai as genai

# Configure API
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No API key found!")
    exit(1)

genai.configure(api_key=api_key)

print("Available models:")
print("-" * 50)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
    print("\nTrying basic models...")
    
    # Test common models
    test_models = [
        "gemini-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-pro",
        "models/gemini-1.5-flash",
    ]
    
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say hi")
            print(f"✓ {model_name} - WORKS!")
            break
        except Exception as e:
            print(f"✗ {model_name} - {str(e)[:80]}")