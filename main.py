import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torcheval.metrics import Perplexity as Perplexity
from torch.utils import checkpoint

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(rank=rank, world_size=world_size, backend='nccl')


def cleanup():
    dist.destroy_process_group()


def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
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

    print("GPU: ", rank)
    print("Loading openai-community/gpt2-large")
    # Load the pre-trained openai-community/gpt2-large (teacher)
    teacher_model_name = "openai-community/gpt2-large"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    # teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

    print("Loading gpt2 model")
    # Load the pre-trained "openai-community/gpt2" model (student)
    student_model_name = "openai-community/gpt2"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    # student_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

    print("Loading wikitext dataset")
    # Example: Load a dataset like "wikitext"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=512)

    setup(rank, world_size)

    # Multiple gpus
    torch.cuda.set_device(rank)

    print("Wrapping teacher in DDP")
    teacher_model.to(rank)
    teacher_model = DDP(teacher_model, device_ids=[rank])

    print("Wrapping student in DDP")
    student_model.to(rank)
    student_model = DDP(student_model, device_ids=[rank])

    print("Starting tokenization")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # DataLoader for the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler)

    # Define optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    print("Memory usage (teacher)", teacher_model.get_memory_footprint()/1e6)
    print("Memory usage (student)", student_model.get_memory_footprint()/1e6)

    num_epochs = 6

    print("Starting training")
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        student_model.train()
        teacher_model.eval()  # Teacher model doesn't need gradient updates

        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = input_ids.clone().detach()  # Language modeling, labels are input_ids

            def custom_student_forward(input_ids, attention_mask):
                return student_model(input_ids=input_ids, attention_mask=attention_mask)

            # Forward pass through the student model
            student_outputs = checkpoint.checkpoint(custom_student_forward,input_ids, attention_mask, use_reentrant=False)
            student_logits = student_outputs.logits

            # Forward pass through the teacher model (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            # Calculate distillation loss
            loss = distillation_loss(student_logits, teacher_logits, labels, T=1.2, alpha=0.7)

            # Backward pass
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader)}")

    # Evaluate the student model
    student_model.eval()

    print("Starting evaluation")
    with torch.no_grad():
        for batch in test_dataloader:
            perplexity_metric = Perplexity().to(rank)
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = input_ids.clone().detach()

            # Forward pass through the student model
            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            print("Calculating log probs")
            log_probs = F.log_softmax(outputs.logits, dim=-1).to(rank)
            print("Updating perplexity inputs")
            perplexity_metric.update(log_probs, labels)


        print("Computing perplexity")
        perplexity_score = perplexity_metric.compute()
        print(f"Perplexity: {perplexity_score}")

    print("Saving model")
    # Save the student model and tokenizer
    model_name = student_model_name.replace("/", "-")
    student_model.module.save_pretrained(f"model-{model_name}_epochs-{num_epochs}_temperature-0.7")
    student_tokenizer.save_pretrained(f"model-{model_name}_epochs-{num_epochs}_temperature-0.7")
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()