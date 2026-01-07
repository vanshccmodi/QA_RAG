import requests
import os
import time

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = (
    "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"
)

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


def build_prompt(question, chunks):
    context = "\n".join([c[3] for c in chunks])

    return f"""
You are a strict question answering system.

Answer ONLY using the context below.
If the answer is not present, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""


def call_llm(prompt, max_retries=5, wait_seconds=6):
    for _ in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt},
                timeout=30
            )

            if not response.text:
                time.sleep(wait_seconds)
                continue

            try:
                data = response.json()
            except ValueError:
                time.sleep(wait_seconds)
                continue

            if isinstance(data, dict) and "error" in data:
                if "loading" in data["error"].lower():
                    time.sleep(wait_seconds)
                    continue
                return "I don't know based on the provided context"

            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()

        except requests.exceptions.RequestException:
            time.sleep(wait_seconds)

    return "I don't know based on the provided context"
