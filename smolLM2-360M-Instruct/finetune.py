from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# Local path to the cloned model
MODEL_PATH = "./SmolLM2-360M-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Load training data
dataset = load_dataset(
    "json",
    data_files="bosnian.jsonl",
)["train"]

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./smollm-bosnian",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

# Fine-tune the model
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model("./smollm-bosnian")

print("Fine-tuning complete.")