import requests

# Simple request to TensorZero Gateway using Ollama
response = requests.post(
    "http://localhost:3000/inference",
    headers={"Content-Type": "application/json"},
    json={
        "model_name": "ollama::llama3",
        "input": {
            "messages": [{"role": "user", "content": "What is the capital of Japan?"}]
        },
    },
)

# Print the response
print(response)
print(response.json()["output"]["content"])
