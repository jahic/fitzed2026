from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.5",
    reasoning={
        "effort": "minimal"
    },
    max_output_tokens=100,
    input=[
        {
            "role": "system",
            "content": (
                "You are a professional translator. "
                "Translate the user's text accurately into Bosnian. "
                "Do not add explanations, notes, or commentary. "
                "Return only the translated text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Translate to Bosnian: "
                "“There is no struggle too vast, no odds too overwhelming, "
                "for even should we fail - should we fall - "
                "we will know that we have lived.”"
            ),
        },
    ],
)

print(response.output_text)