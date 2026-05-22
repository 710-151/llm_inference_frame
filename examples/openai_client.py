from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Chat Completion（自动使用模型的 chat template）
response = client.chat.completions.create(
    model="model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "什么是快速排序？"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "用Python写一个快速排序"}],
    stream=True,
    max_tokens=256,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)