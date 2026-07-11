from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=".",
)

messages = [
    {
        "role": "user",
        "content": "Translate to Bosnian: There is no struggle too vast."
    }
]

response = pipe(
    messages,
    max_new_tokens=10000,
)

print("Response: " + response[0]["generated_text"][-1]["content"])