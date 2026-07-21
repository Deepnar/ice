import requests

payload = {
    "model": "gemma4:12b-256k",
    "messages": [
        {"role": "user", "content": "Rate the following AI answer on a scale of 1 to 5. 5 means perfect.\n\nQuestion: What is 2+2?\n\nAI answer: 4\n\nRating (1-5):"}
    ],
    "temperature": 0.0,
    "max_tokens": 20
}

resp = requests.post("http://localhost:11434/v1/chat/completions", json=payload)
print(resp.status_code)
print(resp.json()["choices"][0]["message"]["content"])