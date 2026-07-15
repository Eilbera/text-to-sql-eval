from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

resp = client.chat.completions.create(
    model="qwen3.5:27b",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print(resp.choices[0].message.content)