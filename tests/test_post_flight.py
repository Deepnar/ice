# tests/test_post_flight.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workers.post_flight import is_lossless, generate_summary

# Test lossless detection
sample_response = "Here is a code snippet: ```print('hello')```"
print(f"Lossless (code): {is_lossless(sample_response)}")  # Should be True

sample_response2 = "Thank you for your question. I appreciate your input."
print(f"Lossless (plain): {is_lossless(sample_response2)}")  # Should be False

# Test summary generation (requires background model running on port 8002)
summary = generate_summary("What is vLLM?", "vLLM is an open-source library for LLM serving.")
print(f"Summary: {summary}")