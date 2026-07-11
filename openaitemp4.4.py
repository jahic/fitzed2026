from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    seed=12345,
    temperature=0,
    max_tokens=100,
    messages=[
        {
            "role": "system",
            "content": "You are a professional translator. Translate the user's text accurately into Bosnian without adding explanations."
        },
        {
            "role": "user",
            "content": "Translate to Bosnian: “There is no struggle too vast, no odds too overwhelming, for even should we fail - should we fall - we will know that we have lived.”"
        }
    ]
)

print(response.choices[0].message.content)
print("System fingerprint:", response.system_fingerprint)