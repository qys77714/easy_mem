from openai import OpenAI

llm = OpenAI(
    # model="gpt-4o-mini",
    api_key="CDHhGET6CaPgEj6e2itV3xQjO10kVyquZaVW7FJvBlu1j4SYHjc0JQQJ99BKACYeBjFXJ3w3AAAAACOGyMor",
    base_url="https://foundationmodeldepartment.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
)

response = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
)

print(response.choices[0].message.content)