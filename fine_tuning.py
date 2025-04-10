import nltk
import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
import evaluate
import numpy as np

# Define file name and such
model_name = "openai-community-gpt2"
distill_amount_of_epochs = "6"
fine_tune_amount_of_epochs = "1"

# Path to the trained model/tokenizer
model_path = f"model-{model_name}_epochs-{distill_amount_of_epochs}_temperature-1.2"
tokenizer_path = f"model-{model_name}_epochs-{distill_amount_of_epochs}_temperature-1.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

# Load the model
#model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=bnb_config)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
model.config.pad_token_id = model.config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print("Memory used (MBs):", model.get_memory_footprint()/1e6)
print(model)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r = 8, # LoRA rank (higher = more aggressive)
    lora_alpha = 16,
    bias =  "none",
    task_type = "CAUSAL_LM",
    #target_modules=['c_attn', 'c_proj', 'c_fc', 'c_proj'],
)

model = get_peft_model(model, peft_config)

# Set the device
print("Moving to gpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loading wikitext dataset")
# Example: Load a dataset like "wikitext"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True,
                             max_length=512)

print("Starting tokenization")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

rouge = evaluate.load("rouge")

def compute_rouge():
    print("Computing ROUGE")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value * 100 for key, value in result.items()}

# SFTTrainer config
training_args = TrainingArguments(
    ## GROUP 1: Memory usage
    # Checkpointing
    gradient_checkpointing=True,  # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False},
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=2,
    auto_find_batch_size=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    ## GROUP 3: These are typical training parameters
    num_train_epochs=2,
    learning_rate=5e-5,
    #max_seq_length=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_stragegy="epoch",

    ## GROUP 4: Logging parameters
    logging_strategy="steps",
    logging_steps=10,
    logging_dir='./logs',
    output_dir=f'./{model_path}-fine_tuning',
    report_to='none'
)

print("Creating trainer")
# Create the trainer
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_rouge,
    args=training_args,
)

print("Starting training")
trainer.train()

print("Training finished, saving model")
trainer.save_model(f'{model_path}-fine_tuning-{fine_tune_amount_of_epochs}')
tokenizer.save_pretrained(f'{model_path}-fine_tuning-{fine_tune_amount_of_epochs}')