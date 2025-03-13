from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
import math

# Load the pre-trained LLaMA 3-8B model (teacher)
teacher_model_name = "meta-llama/Llama-3-8b"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

# Load the pre-trained LLaMA 3-1B model (student)
student_model_name = "meta-llama/Llama-3-1b"
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

def distillation_loss(student_logits, teacher_logits, true_labels, T=2.0, alpha=0.7):
    """
    Computes the knowledge distillation loss.

    :param student_logits: Output logits from the student model
    :param teacher_logits: Output logits from the teacher model
    :param true_labels: Ground truth labels
    :param T: Temperature (scaling factor for softening logits)
    :param alpha: Weight for combining cross-entropy and KL divergence losses
    """
    # Cross entropy loss for the student model on the true labels
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), true_labels.view(-1))

    # Soft targets (teacher's soft output)
    soft_teacher_output = F.softmax(teacher_logits / T, dim=-1)
    soft_student_output = F.log_softmax(student_logits / T, dim=-1)

    # KL Divergence loss
    kl_loss = F.kl_div(soft_student_output, soft_teacher_output, reduction='batchmean')

    # Combine losses with weighting
    return alpha * ce_loss + (1 - alpha) * (T * T) * kl_loss




# Example: Load a dataset like "wikitext"
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_dataset = dataset["train"]

# Tokenize the dataset
def tokenize_function(examples):
    return teacher_tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])



# DataLoader for the dataset
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define optimizer for the student model
optimizer = AdamW(student_model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multiple gpus
teacher_model = nn.DataParallel(teacher_model)
student_model = nn.DataParallel(student_model)

# Move to device
teacher_model.to(device)
student_model.to(device)

num_epochs = 3

for epoch in range(num_epochs):
    student_model.train()
    teacher_model.eval()  # Teacher model doesn't need gradient updates

    total_loss = 0
    for batch in train_dataloader:
        # Move tensors to the correct device (GPU/CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()  # Language modeling, labels are input_ids

        # Forward pass through the student model
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Forward pass through the teacher model (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Calculate distillation loss
        loss = distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader)}")

# Evaluate the student model
student_model.eval()

with torch.no_grad():
    eval_loss = 0
    num_eval_batches = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        eval_loss += outputs.loss.item()
        num_eval_batches += 1

    avg_eval_loss = eval_loss / num_eval_batches
    perplexity = math.exp(avg_eval_loss)
    print(f"Perplexity: {perplexity}")

# Save the student model and tokenizer
student_model.save_pretrained("student_model")
student_tokenizer.save_pretrained("student_model")