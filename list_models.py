import google.generativeai as genai
import os

# Using the key from your test_gemini.py
genai.configure(api_key="AIzaSyBO05CVIq7KH1AnRazYM8dM0H-96ZPhjXw")

print("--- Available Models ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"ID: {m.name}")
