"""Reproduce the qwen3.5:27b baseline via Ollama (running on this Spark)."""
import requests

from common import BASE, PROMPT, grade, load_questions


def ask_qwen(schema, question):
    prompt = PROMPT.format(schema=schema, question=question)
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen3.5:27b",
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"num_ctx": 8192, "temperature": 0},
    }, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


if __name__ == "__main__":
    ds = load_questions()
    grade(ds, ask_qwen, BASE + "/results/score_27b_ollama.json", "27b")
