import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define file name and such
model_name = "openai-community-gpt2"
amount_of_epochs = "6"

# Path to the trained model/tokenizer
model_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2"
tokenizer_path = f"model-{model_name}_epochs-{amount_of_epochs}_temperature-1.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model.get_memory_footprint()/1e6)
print(model)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    bias =  "none",
    task_type = "CAUSAL_LM",
    target_modules=['c_attn', 'c_proj', 'c_fc', 'c_proj'],
)

model = get_peft_model(model, config)

print("Loading wikitext dataset")
# Example: Load a dataset like "wikitext"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# SFTTrainer config
sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,  # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False},
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=16,
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=True,

    ## GROUP 2: Dataset-related
    max_seq_length=64,
    # Dataset
    # packing a dataset means no padding is needed
    packing=True,

    ## GROUP 3: These are typical training parameters
    num_train_epochs=10,
    learning_rate=3e-4,
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='paged_adamw_8bit',

    ## GROUP 4: Logging parameters
    logging_steps=10,
    logging_dir='./logs',
    output_dir=f'./{model_path}-fine_tuning',
    report_to='none'
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
trainer.save_model(f'{model_path}-fine_tuning')