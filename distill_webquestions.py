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


def new_distillation_loss(alpha, beta,  student, teacher, tokenizer, embedder, gen_config, rank, batch):    
        with torch.no_grad():
            teacher_outputs = teacher.module.generate(**batch["input_ids"], generation_config=gen_config)

        teacher_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in teacher_outputs]
        teacher_inputs = tokenizer(teacher_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(rank)

        student_logits = student(teacher_inputs.input_ids).logits
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_inputs.input_ids[..., 1:].contiguous()
        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]
        loss_ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        with torch.no_grad():
            teacher_embeddings = embedder.encode(teacher_texts, convert_to_tensor=True).to(rank)

        student_generated = student.module.generate(**batch["input_ids"], generation_config=gen_config)
        student_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in student_generated]
        student_embeddings = embedder.encode(student_texts, convert_to_tensor=True).to(rank)
        loss_embed = F.mse_loss(student_embeddings, teacher_embeddings)
        student_free_inputs = tokenizer(student_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(rank)
        free_logits = student(student_free_inputs.input_ids).logits
        shift_free_logits = free_logits[..., :-1, :].contiguous()
        shift_teacher_labels = teacher_inputs.input_ids[..., 1:].contiguous()
        min_len2 = min(shift_free_logits.size(1), shift_teacher_labels.size(1))
        shift_free_logits = shift_free_logits[:, :min_len2, :]
        shift_teacher_labels = shift_teacher_labels[:, :min_len2]

        loss_consistency = F.cross_entropy(shift_free_logits.reshape(-1, shift_free_logits.size(-1)), shift_teacher_labels.reshape(-1))

        total_loss = loss_ce + alpha * loss_embed + beta * loss_consistency
        return total_loss

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
    
    embedder = AutoModelForCausalLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    gen_config = {
        "repetition_penalty": 1.2,
    }

    print("Loading wikitext dataset")
    # Example: Load a dataset like "wikitext"
    dataset = load_dataset("web_questions")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = teacher_tokenizer(examples['question'], return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=128)
        
        # Answers preprocessing
        if isinstance(examples['answers'], list):
            examples["answers"] = [str(text) for text in examples["answers"]]
        else:
            examples["answers"] = [str(examples["answers"])]
        labels = teacher_tokenizer(examples['answers'], return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=128)
        inputs['labels'] = labels['input_ids']
        return inputs

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

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # DataLoader for the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler)

    # Define optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    num_epochs = 10

    print("Starting training")
    for epoch in range(num_epochs):
        alpha = 0.5
        beta = 0.5
        train_sampler.set_epoch(epoch)
        student_model.train()
        teacher_model.eval()  # Teacher model doesn't need gradient updates

        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)

            # Calculate distillation loss
            loss = new_distillation_loss(alpha, beta, student_model, teacher_model, teacher_tokenizer, embedder, gen_config, rank, batch)

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
            labels = batch["labels"].to(rank)

            # Forward pass through the student model
            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
    student_model.module.save_pretrained(f"model-{model_name}_epochs-{num_epochs}_webquestions_alpha-{alpha}_beta-{beta}")
    student_tokenizer.save_pretrained(f"model-{model_name}_epochs-{num_epochs}_-webquestions_alpha-{alpha}_beta-{beta}")
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()