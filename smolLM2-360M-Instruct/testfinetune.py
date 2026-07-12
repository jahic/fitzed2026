import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "./SmolLM2-360M-Instruct"
ADAPTER_PATH = "./smollm-bosnian"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

messages = [
    {
        "role": "system",
        "content": "You are a professional translator into Bosnian."
    },
    {
        "role": "user",
        "content": "Translate to Bosnian: Struggle."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(response)