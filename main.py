import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
import math
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(rank=rank, world_size=world_size, backend='nccl')


def cleanup():
    dist.destroy_process_group()


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


def train(rank, world_size):
    login(os.getenv("HF_TOKEN"))

    print("Loading Qwen2.5-1.5B model")
    # Load the pre-trained Qwen/Qwen2.5-0.5B model (teacher)
    teacher_model_name = "Qwen/Qwen2.5-1.5B"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    # teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

    print("Loading Qwen/Qwen2.5-1.5B model")
    # Load the pre-trained "Qwen/Qwen2.5-1.5B" model (student)
    student_model_name = "Qwen/Qwen2.5-0.5B"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    # student_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

    print("Loading wikitext dataset")
    # Example: Load a dataset like "wikitext"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=512)

    setup(rank, world_size)

    # Set up the environment for distributed training
    print("Wraps in DataParallel container")
    # Multiple gpus
    torch.cuda.set_device(rank)

    teacher_model.to(rank)
    teacher_model = DDP(teacher_model, device_ids=[rank])

    student_model.to(rank)
    student_model = DDP(student_model, device_ids=[rank])

    print("Starting tokenization")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    # DataLoader for the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    # Define optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    # Training loop
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("No GPU available")

    # teacher_model = nn.DataParallel(teacher_model)
    # student_model = nn.DataParallel(student_model)

    print("Moving to GPU")
    # Move to device
    # teacher_model.to(device)
    # student_model.to(device)

    print("Memory usage", torch.cuda.memory_summary())

    num_epochs = 3

    print("Starting training")
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        student_model.train()
        teacher_model.eval()  # Teacher model doesn't need gradient updates

        total_loss = 0
        for batch in train_dataloader:

            input_ids = batch["input_ids"].to(device)
            print("input ids:", input_ids.shape)
            attention_mask = batch["attention_mask"].to(device)
            print("attention mask:", attention_mask.shape)
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
    cleanup()
    student_model.eval()

    print("Starting evaluation")
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

    print("Saving model")
    # Save the student model and tokenizer
    student_model.save_pretrained("student_model")
    student_tokenizer.save_pretrained("student_model")


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()