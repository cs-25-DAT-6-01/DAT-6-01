import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
import torch.distributed as dist
from torcheval.metrics import Perplexity as Perplexity
from torch.utils import checkpoint
from sentence_transformers import SentenceTransformer
from utility import plot_metrics
from utility import filter_lines

from loss_functions import prototype_log_loss


def train():
    # Login to Hugging Face Hub
    login(os.getenv("HF_TOKEN"))

    print("Loading openai-community/gpt2-large")
    # Load the teacher model (large GPT-2)
    teacher_model_name = "openai-community/gpt2-large"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    # Set the pad token to the end-of-sequence token if not already set
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    # Set the padding side to left for the tokenizer
    teacher_tokenizer.padding_side = "left"
    # Load model with automatic device mapping and dtype
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, device_map="auto", torch_dtype="auto"
    )

    print("Loading gpt2 model")
    # Load the pre-trained "openai-community/gpt2" model (student)
    student_model_name = "openai-community/gpt2"
    # Load the tokenizer for the student model
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    # Set the pad token to the end-of-sequence token if not already set
    student_tokenizer.pad_token = student_tokenizer.eos_token
    # Set the padding side to left for the tokenizer
    student_tokenizer.padding_side = "left"
    # Load the student model with automatic device mapping and dtype
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name, device_map="auto", torch_dtype="auto"
    )

    print("Loading wikitext dataset")
    # Load the wikitext dataset
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty lines and limit the dataset size for faster training
    train_dataset = train_dataset.map(
        lambda example: {"text": filter_lines(example["text"])}
    )
    # Filter out empty examples
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0)
    # Limit the dataset size for faster training
    train_dataset = train_dataset.select(range(10000))

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    print("Starting tokenization")
    # Tokenize the training dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    # Set the format for the dataset to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # DataLoader for the dataset
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, num_workers=4, pin_memory=True
    )

    # Get the first device for both models
    student_first_device = list(student_model.hf_device_map.values())[0]
    teacher_first_device = list(teacher_model.hf_device_map.values())[0]

    # Define optimizer for the student model
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    num_epochs = 10

    # Plotting metrics
    loss_history = []
    ppl_history = []

    print("Starting training")
    for epoch in range(num_epochs):
        # Distillation configuration parameters
        alpha = 8
        lambd = 0.7
        beta = 0.3
        gamma = 1.5
        temperature = 2
        student_model.train()

        total_loss = 0
        epoch_start = time.time()

        # Iterate over the training dataloader
        for step, batch in enumerate(train_dataloader):
            # Move the perplexity metric to the student's first device
            perplexity_metric = Perplexity().to(student_first_device)

            # Move input_ids and attention_mask to the student's first device
            input_ids = batch["input_ids"].to(student_first_device)
            attention_mask = batch["attention_mask"].to(student_first_device)
            labels = input_ids.clone().detach()

            # Compute the teacher's logits without gradients
            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids.to(teacher_first_device),
                    attention_mask=attention_mask.to(teacher_first_device),
                ).logits

            # Compute the student's logits
            student_logits = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # Compute the distillation loss components
            # Using the prototype_log_loss function to compute the loss and its components
            loss, kl, align, toptok, entropy = prototype_log_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_first_device=student_first_device,
                return_components=True,
            )

            # Every 100 steps, print the loss components
            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}: "
                    f"Loss={loss.item():.2f} | KL={kl.item():.2f} | Align={align.item():.2f} | "
                    f"TopTok={toptok.item():.4f} | Entropy={entropy.item():.2f}"
                )

            # Zero the gradients, perform backpropagation, and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the perplexity metric with the student's log probabilities
            log_probs = F.log_softmax(student_logits, dim=-1)
            perplexity_metric.update(log_probs, labels)
            total_loss += loss.item()

        # Compute the average loss for the epoch and the time taken for the epoch
        epoch_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start
        perplexity_score = perplexity_metric.compute()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Perplexity: {perplexity_score}, Time: {epoch_time:.2f}s"
        )

        # Append the epoch loss and perplexity score to the history for plotting
        loss_history.append(epoch_loss)
        ppl_history.append(perplexity_score)

    print("Saving model")
    # Save the student model and tokenizer
    model_name = student_model_name.replace("/", "-")
    out_dir = f"model-{model_name}_epochs-{num_epochs}_wikitext_alpha-{alpha}_beta-{beta}_lamb-{lambd}_gam-{gamma}_temp-{temperature}"

    student_model.save_pretrained(out_dir)
    student_tokenizer.save_pretrained(out_dir)

    # Plot the training metrics
    plot_metrics(
        metrics={"loss": loss_history, "perplexity": ppl_history},
        run_tag="wikitext",
        out_dir=out_dir,
    )


def main():
    train()


if __name__ == "__main__":
    main()
