# save as test_ollama.py and run: python test_ollama.py
import requests
print(requests.get("http://127.0.0.1:11434/api/tags", timeout=5).status_code)
